import torch.nn as nn

# Exemple MLP implementation
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