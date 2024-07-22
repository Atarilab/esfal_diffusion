import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, repeat
from .DDPM import DDPM

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device)
        emb = 1 / (10000 ** (2 * (emb // 2) / half_dim))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalMLPBase(nn.Module):
    def __init__(self,
                 input_dim,
                 diffusion_step_embed_dim:int=32,
                 hidden_dim=128,
                 num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = input_dim

        self.pose_embedding = SinusoidalPosEmb(diffusion_step_embed_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

        # Network to process conditioning data
        self.cond_fc1 = nn.Linear(91*3 + diffusion_step_embed_dim, hidden_dim)
        self.cond_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.GELU()

    def forward(self, x, t, conditioning_data):
        
        batch_size, T, D = x.shape
        x = rearrange(x, 'b t d -> b (t d)')

        # Process conditioning data
        conditioning_data = conditioning_data.reshape(batch_size, -1)  # Flatten
        t = self.pose_embedding(t)
        conditioning_data = torch.cat((conditioning_data, t), dim=-1)
        conditioning_data = self.activation(self.cond_fc1(conditioning_data))
        conditioning_data = self.activation(self.cond_fc2(conditioning_data))
        # Combine conditioning data with x using FiLM (Feature-wise Linear Modulation)
        scale = conditioning_data + 1
        shift = conditioning_data

        x = self.activation(self.fc1(x))
        for fc in self.fc_layers:
            x = self.activation(fc(x)) * scale + shift
        x = self.fc_out(x)
        x = rearrange(x, 'b (t d) -> b t d', t=T, d=D)
        return x
    

class ConditionalMLP(DDPM):
    def __init__(self,
                 input_dim,
                 diffusion_step_embed_dim:int=32,
                 hidden_dim=128,
                 num_layers=4,
                 **kwargs):
        model = ConditionalMLPBase(
            input_dim,
            diffusion_step_embed_dim,
            hidden_dim,
            num_layers
        )
        super().__init__(model, **kwargs)