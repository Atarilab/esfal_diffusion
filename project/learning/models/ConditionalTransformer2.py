import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
try:
    from .DDPM import DDPM
    from .CDCD import CDCD
except:
    from models.DDPM import DDPM
    from models.CDCD import CDCD

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
    
class FiLM(nn.Module):
    def __init__(self, seq_length, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.avg_seq = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Linear(seq_length, 1, bias=False),
            Rearrange("b c t -> b t c"),
        )
        self.film_cond = nn.Sequential(
            self.MLP,
            self.avg_seq,
        )
    
    def forward(self, x, conditioning):
        scale, bias = torch.split(self.film_cond(conditioning), [self.hidden_dim, self.hidden_dim], dim=-1)
        x = scale * x + bias
        return x


class CDCDTransformerConditionedBase2(nn.Module):
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
            nn.Linear(3, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Mish(),
        )

        self.x_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Mish(),
        )

        self.state_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Mish(),
        )

        self.pose_embedding = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.transformer_blocks = nn.ModuleList([
                TransformerBlockConditioned(hidden_dim, hidden_dim, dropout, num_heads, need_weights=(l == n_layers-1))
                for l in range(n_layers)
            ])
        
        self.film_blocks = nn.ModuleList([
                FiLM(11, hidden_dim)
                for l in range(n_layers)
            ])
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, 
                                                                nhead=num_heads,
                                                                dim_feedforward=hidden_dim,
                                                                dropout=dropout,
                                                                batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=n_layers)
        
        self.V_logits = nn.Linear(hidden_dim, output_length, bias=True)

    def get_last_attention(self):
        return self.transformer_blocks[-1].attn_weights

    def forward(self, x, t, condition, self_conditioning=None, return_logits:bool=False):
        B = x.shape[0]

        # Get locations to select, state variables
        condition = condition.reshape(condition.shape[0], -1)
        condition = rearrange(condition, "b (l c) -> b l c", c = 3)
        state, locations = torch.split(condition, [self.exclude_first, condition.shape[1] - self.exclude_first], dim=1)
        locations_copy = deepcopy(locations)
        x_copy = x

        # Embedding
        state = self.state_embedding(state)
        locations = self.contact_embedding(locations)
        x = self.x_embedding(x)
        t = self.pose_embedding(t).unsqueeze(-2)
        if self_conditioning != None:
            self_conditioning = self.contact_embedding(self_conditioning)
            x = torch.cat((x, self_conditioning), dim=1)

        t_state = torch.cat((t, state), dim=1)

        # Encoder
        # Cross attention transformer
        for block, film in zip(self.transformer_blocks, self.film_blocks):
            locations = block(locations, x)
            locations = film(locations, t_state)

        # Logits
        logits = self.V_logits(locations).squeeze(-1)

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
            return logits, expectation

        return expectation

class CDCDTransformerConditioned2(CDCD):
    def __init__(self,
                output_length:int,
                exclude_first:int,
                hidden_dim:int,
                n_layers:int,
                num_heads:int=-1,
                dropout:float=0.1,
                **kwargs,
                ) -> None:
    
        model = CDCDTransformerConditionedBase2(
            output_length,
            exclude_first,
            hidden_dim,
            n_layers,
            num_heads,
            dropout
        )

        super().__init__(model, **kwargs)

if __name__ == "__main__":
    H = 64
    N_LAYERS = 4
    DROPOUT = 0.1
    OUTPUT_LENGTH = 8
    XCLUDE = 10
    model = TransformerConditioned(OUTPUT_LENGTH, XCLUDE, H, N_LAYERS, DROPOUT)

    dummy_input = torch.rand(2, 91, 3)
    out = model(dummy_input)

    print(out.shape)
