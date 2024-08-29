

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

class MLP(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 n_hidden:int=1,
                 latent_dim:int=32,
                 ) -> None:
        super(MLP, self).__init__()

        layers = [nn.Linear(input_dim, latent_dim), nn.LeakyReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(latent_dim, latent_dim), nn.LeakyReLU()]
        layers += [nn.Linear(latent_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.film_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
        )
    
    def forward(self, x, conditioning):
        scale, bias = torch.split(self.film_cond(conditioning), [self.hidden_dim, self.hidden_dim], dim=-1)
        x = scale * x + bias
        return x

class ConditionalMLP(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 cond_dim:int,
                 n_hidden:int,
                 latent_dim:int,
                 ) -> None:
        super(ConditionalMLP, self).__init__()

        # First and last layer of the conditional MLP
        self.in_layer = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LeakyReLU())
        self.out_layer = nn.Linear(latent_dim, output_dim)

        # MLP layers with FiLM conditioning
        mlp_layers = []
        film_layers = []
        norm_layers = []
        for _ in range(n_hidden):
            mlp_layers += [nn.Linear(latent_dim, latent_dim), nn.LeakyReLU()]
            film_layers += [FiLM(cond_dim, latent_dim)]
            norm_layers += [nn.LayerNorm(latent_dim)]

        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.film_layers = nn.Sequential(*film_layers)
        self.norm_layers = nn.Sequential(*norm_layers)

    def forward(self, x, condition):
        x = self.in_layer(x)
        condition = condition.reshape(x.shape[0], 1, -1)
        for mlp, film, norm in zip(self.mlp_layers,
                                   self.film_layers,
                                   self.norm_layers):
            x = norm(film(mlp(x), condition) + x)

        x = self.out_layer(x)
        return x

class ConditionalMLP_CDCD(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 n_hidden:int,
                 latent_dim:int,
                 exclude_first:int,
                 diffusion_step_embed_dim:int=32,
                 **kwargs,
                 ) -> None:
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super(ConditionalMLP_CDCD, self).__init__()
        
        self.exclude_first = exclude_first
        pos_embedding_dim = output_dim // 2
        cond_dim = output_dim * (pos_embedding_dim + input_dim) + diffusion_step_embed_dim + exclude_first * input_dim

        self.mlp_cond = ConditionalMLP(input_dim,
                                       output_dim,
                                       cond_dim,
                                       n_hidden,
                                       latent_dim)

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 2),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 2, diffusion_step_embed_dim),
        )

        self.points_index_encoder = nn.Sequential(
            SinusoidalPosEmb(pos_embedding_dim),
            nn.Linear(pos_embedding_dim, pos_embedding_dim * 2),
            nn.Mish(),
            nn.Linear(pos_embedding_dim * 2, pos_embedding_dim),
        )
        
        self.softmax = nn.Softmax(-1)
        self.pointers = None
        self.max_probs = None

    def forward(self,
                noisy_data_samples: torch.Tensor,
                timestep: torch.Tensor=None,
                global_cond=None,
                return_logits:bool=False,
                **kwargs):
        """
        noisy_data_samples: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        
        B, T, D = noisy_data_samples.shape

        timestep_cond = self.diffusion_step_encoder(timestep) # (B, H_pos)

        # State variables (exclude_first, D) and contact locations (N, D)
        state, cnt_loc = torch.split(global_cond, [self.exclude_first, global_cond.shape[1] - self.exclude_first], dim=1)
        cnt_loc = cnt_loc.reshape(B, -1, D)
        state = state.reshape(B, -1)
        
        d = noisy_data_samples.device

        cnt_loc_index = torch.arange(cnt_loc.shape[1]).expand(B, -1).unsqueeze(-1).to(d)
        cnt_loc_index = self.points_index_encoder(cnt_loc_index).squeeze()
        cnt_loc_cond = torch.cat([cnt_loc, cnt_loc_index], dim=-1) # (B, N, D + H_pos), position embedding
        cnt_loc_cond = cnt_loc_cond.reshape(B, -1) # (B, N * (D + H_pos))
        
        conditioning = torch.cat([
            timestep_cond, cnt_loc_cond, state
        ], axis=-1) # (B, N * (D + H_pos) + H_pos + exclude_first * D)
        # (B, T, N)\
        # Change the temperature according to the diffusion step, force to select one of the element
        logits = self.mlp_cond(noisy_data_samples, conditioning) * 1 / timestep.unsqueeze(-1).unsqueeze(-1)
        logits = torch.swapaxes(logits, -1, -2)

        # Compute expectation
        probs = torch.nn.functional.softmax(logits, dim=1).unsqueeze(dim=-1) # (B, T, N, 1)
        expectation = torch.sum(probs * cnt_loc.unsqueeze(dim=-2), dim=1) # (B, T, 2)
        self.max_probs, self.pointers = torch.max(probs, dim=1, keepdim=True)

        if return_logits:
            return logits, expectation

        return expectation   

if __name__ == "__main__":
    B = 32
    H = 64
    D = 3 # 3D points
    T = 8 # Number of diffusion points
    N = 20 # Number of conditioning 3D points
    N_STATE = 14 # N state variables
    N_LAYERS = 4
    INPUT_DIM = D
    OUTPUT_DIM = N
    
    model = ConditionalMLP_CDCD(INPUT_DIM, OUTPUT_DIM, N_LAYERS, H, N_STATE)
    
    dummy_input = torch.rand(B, T, D)
    dummy_loc =  torch.rand(B, D * 20)
    dummy_state =  torch.rand(B, D * N_STATE)
    dummy_cond = torch.cat((dummy_state, dummy_loc), dim=-1)
    dummy_t = torch.randint(0, 100, (B,))
    logits, expectations = model(dummy_input, dummy_t, dummy_cond, return_logits = True)

    print(logits.shape, expectations.shape)