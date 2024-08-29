import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.A = torch.log(torch.tensor(10000))

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = self.A / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * - emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class TransformerBlockConditioned(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dim:int,
                 dropout:float=0.1,
                 num_heads:int=-1,
                 need_weights:bool=False,
                 ):
        super().__init__()
        self.need_weights = need_weights
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim,
                                               num_heads=min(input_dim//8, 8) if num_heads < 0 else num_heads,
                                               dropout=dropout,
                                               batch_first=True,)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.attn_weights = None

    def forward(self, x, condition):
        x = self.norm1(x + self._sa_block(x, condition))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, condition):
        if self.need_weights:
            att, self.attn_weights = self.self_attn(x, condition, condition, need_weights=self.need_weights, average_attn_weights=False)
        else:
            att = self.self_attn(x, condition, condition, need_weights=self.need_weights)[0]
        return att
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

class TransformerConditioned(nn.Module):
    def __init__(self,
                 output_length:int,
                 exclude_first:int,
                 hidden_dim:int,
                 n_layers:int,
                 num_heads:int=-1,
                 dropout:float=0.1,
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.exclude_first = exclude_first

        self.inputs_embedding = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(3, hidden_dim // 2, kernel_size=3, padding=1, stride=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.Mish(),
            Rearrange("b h t -> b t h"),
        )
        self.transformer_blocks = nn.ModuleList([
                TransformerBlockConditioned(hidden_dim, hidden_dim, dropout)
                for l in range(n_layers)
            ])

        self.last_attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                    num_heads=min(hidden_dim//8, 8) if num_heads < 0 else num_heads,
                                                    dropout=dropout,
                                                    batch_first=True,)
        
        self.V = nn.Linear(hidden_dim, output_length, bias=False)

        self.pointers = None
        self.attn_weights = None


    def forward(self, input):
        #B = x.shape[0]
        #x_copy = deepcopy(x)
        B, L, _ = input.shape
        condition, x = torch.split(input, [self.exclude_first, L - self.exclude_first], dim=1)
        x = self.inputs_embedding(x)
        condition = self.inputs_embedding(condition)

        for block in self.transformer_blocks:
            x = block(x, condition)

        x, self.attn_weights = self.last_attention(x, condition, condition, average_attn_weights=False)
        logits = self.V(x).squeeze(-1)
        self.pointers = torch.argmax(logits, dim=1, keepdim=True).squeeze(1) + self.exclude_first
        logits = torch.cat((torch.zeros((B, self.exclude_first, logits.shape[-1])), logits), dim=1)

        return logits

    def select(self, inputs):
        """
        Inputs:
            - inputs: Input sequence [B, Li, C]
        Returns:
            - outputs: elements from the input sequence [B, Lo, C]
        """
        logits = self.forward(inputs)
        max_probs = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        outputs = torch.take_along_dim(inputs, self.pointers.unsqueeze(-1), dim=1)
        return outputs, max_probs


class CDCDTransformerConditionedBase(nn.Module):
    def __init__(self,
                 output_length:int,
                 exclude_first:int,
                 hidden_dim:int,
                 n_layers:int,
                 num_heads:int=-1,
                 dropout:float=0.1,
                 temperature:float=1.,
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_length = output_length
        self.exclude_first = exclude_first
        self.temperature = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.pointers = None
        self.selected = None

        self.contact_embedding = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(3, hidden_dim * 2, kernel_size=3, padding=1, stride=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.Mish(),
            Rearrange("b h t -> b t h"),
        )

        self.state_embedding = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(3, hidden_dim * 2, kernel_size=3, padding=1, stride=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.Mish(),
            Rearrange("b h t -> b t h"),
        )

        self.x_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish()
        )

        self.pose_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.embedding_transformer = TransformerBlockConditioned(3, hidden_dim, dropout=0., num_heads=1)

        self.transformer_blocks = nn.ModuleList([
                TransformerBlockConditioned(hidden_dim, hidden_dim, dropout, num_heads, need_weights=(l == n_layers-1))
                for l in range(n_layers)
            ])
        
        self.V_logits = nn.Linear(hidden_dim, output_length, bias=False)

    def get_last_attention(self):
        return self.transformer_blocks[-1].attn_weights

    def forward(self, x, t, condition, self_conditioning=None, return_logits:bool=False):
        B = x.shape[0]

        # Get locations to select, state variables
        condition = condition.reshape(condition.shape[0], -1)
        condition = rearrange(condition, "b (l c) -> b l c", c = 3)
        state, locations = torch.split(condition, [self.exclude_first, condition.shape[1] - self.exclude_first], dim=1)
        goal = condition[:, 4:8, :]
        locations_copy = locations
        
        # Embedding
        state = self.state_embedding(state)
        x = self.x_embedding(x)
        goal = self.contact_embedding(goal)
        state[:, 4:8, :] = goal
        locations = self.contact_embedding(locations)
        t = self.pose_embedding(t).unsqueeze(-2)
        if self_conditioning != None:
            self_conditioning = self.x_embedding(self_conditioning)
            t_state_x = torch.cat((t, state, x, self_conditioning), dim=1)
        else:
            t_state_x = torch.cat((t, state, x), dim=1)


        # Cross attention transformer
        for block in self.transformer_blocks:
            locations = block(locations, t_state_x)
        
        # Logits
        logits = self.V_logits(locations)

        # Compute expectation
        probs = torch.nn.functional.softmax(logits, dim=1).unsqueeze(dim=-1)
        expectation = torch.sum(probs * locations_copy.unsqueeze(dim=-2), dim=1)

        # Shift predictions to match the whole sequence
        fill_logits = torch.full((B, self.exclude_first, self.output_length), -torch.inf)
        logits = torch.cat((fill_logits, logits), dim=1)
        self.max_probs, self.pointers = torch.max(probs, dim=1, keepdim=True)
        self.pointers = self.pointers.squeeze(1) + self.exclude_first
        self.selected = torch.take_along_dim(condition, self.pointers, dim=1)

        if return_logits:
            return logits
        
        return expectation

# class CDCDTransformerConditioned(CDCD):
#     def __init__(self,
#                 output_length:int,
#                 exclude_first:int,
#                 hidden_dim:int,
#                 n_layers:int,
#                 num_heads:int=-1,
#                 dropout:float=0.1,
#                 **kwargs,
#                 ) -> None:
    
#         model = CDCDTransformerConditionedBase(
#             output_length,
#             exclude_first,
#             hidden_dim,
#             n_layers,
#             num_heads,
#             dropout
#         )

#         super().__init__(model, **kwargs)

# if __name__ == "__main__":
#     H = 64
#     N_LAYERS = 4
#     DROPOUT = 0.1
#     OUTPUT_LENGTH = 8
#     XCLUDE = 10
#     model = TransformerConditioned(OUTPUT_LENGTH, XCLUDE, H, N_LAYERS, DROPOUT)

#     dummy_input = torch.rand(2, 91, 3)
#     out = model(dummy_input)

#     print(out.shape)
