from typing import Any, Mapping
import torch
import torch.nn as nn


class SNRModelBase(nn.Module):
    """
    Base class for learnable SNR model.
    """
    def __init__(self, num_timesteps:int, learnable:bool=True) -> None:
        super(SNRModelBase, self).__init__()
        self.register_buffer("num_timesteps", torch.tensor(num_timesteps).long())

    def alpha_cum_prod(self, timesteps):
        return torch.nn.functional.sigmoid(- self.forward(timesteps))
    
    def SNR(self, timesteps):
        return torch.exp(- self.forward(timesteps))      

    def SNR_t_prev_sub_SNR_t(self, timesteps):
        """
        SNR(t-1) - SNR(t)
        """
        return self.SNR(timesteps-1) - self.SNR(timesteps)
    
    def SNR_t_prev_div_SNR_t(self, timesteps):
        """
        SNR(t-1) / SNR(t)
        """
        return self.SNR(timesteps-1) / self.SNR(timesteps)
    
    def forward(self, timesteps:torch.IntTensor) -> torch.FloatTensor:
        return timesteps / torch.max(timesteps)[0]
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        snr_model_state_dict = {
            k.split("snr_model.")[-1] : v for k, v in state_dict.items()
            if "snr_model" in k
        }
        super().load_state_dict(snr_model_state_dict, strict, assign)


class SNRModelCumulative(SNRModelBase):
    """
    Monotically increasing NN that represent neg log SNR ratio.
    """
    def __init__(self, num_timesteps:int , trainable:bool=True) -> None:
        super().__init__(num_timesteps, trainable)
        self.base = torch.nn.Parameter(torch.zeros((1,)) , requires_grad=trainable)
        self.W = torch.nn.Parameter(torch.zeros((self.num_timesteps + 1)), requires_grad=trainable)
        self.w_ini, self.base_ini = self.init_weights()

    def init_weights(self):
        """
        Find intial weight value so that alpha_cum_prod(T/2) = 0.5
        """
        SEARCH_SIZE = 100
        w_ini_space = torch.linspace(-10., 10., SEARCH_SIZE)
        base_space = torch.linspace(0., 10., SEARCH_SIZE)

        min_loss = torch.inf

        half_T = self.num_timesteps // 2
        timesteps_center = torch.tensor([half_T - 1, half_T, half_T + 1]).long()
        timesteps_start_end = torch.tensor([0, self.num_timesteps]).long()

        for self.w_ini in w_ini_space:
            for self.base_ini in base_space:
                self.min_diff = self.w_ini / 1000.

                alpha_prod_t_prev, alpha_prod_t, alpha_prod_t_next = torch.split(self.alpha_cum_prod(timesteps_center), 1)
                alpha_prod_start, alpha_prod_end = torch.split(self.alpha_cum_prod(timesteps_start_end), 1)
                
                d_start = torch.abs(alpha_prod_start - 1.)
                d_end = torch.abs(alpha_prod_end - 0.)

                d_to_center = torch.abs(alpha_prod_t - 0.5)

                slope = (alpha_prod_t_next - alpha_prod_t_prev) / 2.
                slope_center = torch.abs(slope)

                loss = (d_to_center +
                        d_start +
                        d_end +
                        slope_center)
                
                if loss < min_loss:
                    min_loss = loss
                    w_ini = self.w_ini
                    base_ini = self.base_ini

        return w_ini, base_ini

    def alpha_cum_prod(self, timesteps):
        return torch.nn.functional.sigmoid(-self.forward(timesteps))
    
    def SNR(self, timesteps):
        return torch.exp(-self.forward(timesteps))      

    def SNR_t_prev_sub_SNR_t(self, timesteps):
        return self.SNR(timesteps-1) - self.SNR(timesteps)
    
    def SNR_t_prev_div_SNR_t(self, timesteps):
        """
        SNR(t-1)/SNR(t) = exp( cumsum(W[:t]) - cumsum(W[:t-1]) ) 
                        = exp( W(t) )
        """
        W = torch.nn.functional.relu(self.W + self.w_ini) + self.min_diff
        W_t = torch.take_along_dim(W, timesteps, dim=-1)
        return torch.exp( W_t )
    
    def get_weights(self, timesteps):
        return torch.take_along_dim(torch.nn.functional.sigmoid(self.W + self.w_ini), timesteps, dim=-1) + self.min_diff
                             
    def forward(self, timesteps):
        """
        SNR(t) = exp( - (cumsum(W[:t]) - slope) )
        Compute (cumsum(W[:t]) - slope)
        """
        W = torch.cumsum(torch.nn.functional.sigmoid(self.W + self.w_ini) + self.min_diff, dim=0)
        W -= (self.base_ini + self.base)
        return torch.take_along_dim(W, timesteps, dim=-1)
    

class SNRModelSigmoid(SNRModelBase):
    """
    Monotically increasing NN that represent neg log SNR ratio.
    """
    def __init__(self, num_timesteps:int, trainable:bool=True) -> None:
        super().__init__(num_timesteps, trainable)
        
        self.alpha = torch.nn.Parameter(torch.zeros((1,)) , requires_grad=trainable)
        self.beta = torch.nn.Parameter(torch.zeros((1,)), requires_grad=trainable)
        self.scale = torch.nn.Parameter(torch.zeros((1,)), requires_grad=trainable)

        self.eps = 1. / self.num_timesteps
        self.alpha_ini, self.beta_ini = self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """
        Find intial weight value so that the alpha prod values 
        """
        SEARCH_SIZE = 100
        alpha_ini_space = torch.linspace(0., 500., SEARCH_SIZE).unsqueeze(-1)
        beta_ini_space = torch.linspace(-100., 0., SEARCH_SIZE).unsqueeze(-1)

        min_loss = torch.inf

        half_T = self.num_timesteps // 2
        timesteps_center = torch.tensor([half_T - 1, half_T, half_T + 1]).long()
        timesteps_start_end = torch.tensor([0, self.num_timesteps]).long()

        for self.alpha_ini in alpha_ini_space:
            for self.beta_ini in beta_ini_space:
                alpha_prod_t_prev, alpha_prod_t, alpha_prod_t_next = torch.split(self.alpha_cum_prod(timesteps_center), 1)
                alpha_prod_start, alpha_prod_end = torch.split(self.alpha_cum_prod(timesteps_start_end), 1)
                
                d_start = torch.abs(alpha_prod_start - 1.)
                d_end = torch.abs(alpha_prod_end - 0.)

                d_to_center = torch.abs(alpha_prod_t - 0.5)

                slope = (alpha_prod_t_next - alpha_prod_t_prev) / 2.
                slope_center = torch.abs(slope)

                loss = (d_to_center +
                        d_start +
                        d_end +
                        slope_center)
                
                if loss < min_loss:
                    min_loss = loss
                    alpha_ini = self.alpha_ini
                    beta_ini = self.beta_ini

        return alpha_ini, beta_ini
                              
    def forward(self, timesteps):
        power = 0.5
        t = timesteps / self.num_timesteps
        t_scale = (t - 0.5) * (1. + self.scale)
        return torch.nn.functional.relu(self.alpha + self.alpha_ini) * torch.nn.functional.sigmoid(t_scale)  ** power + (self.beta + self.beta_ini) + self.eps * t


class SNRModelArsech(SNRModelBase):
    """
    Monotically increasing NN that represent neg log SNR ratio.
    """
    def __init__(self, num_timesteps:int, trainable:bool=True) -> None:
        super().__init__(num_timesteps, trainable)
        
        self.alpha = torch.nn.Parameter(torch.zeros((1,)) , requires_grad=trainable)
        self.beta = torch.nn.Parameter(torch.zeros((1,)), requires_grad=trainable)
        self.scale = torch.nn.Parameter(torch.zeros((1,)), requires_grad=trainable)
        self.shift = torch.nn.Parameter(torch.zeros((1,)), requires_grad=trainable)

        self.eps = 1. / self.num_timesteps
        self.alpha_ini, self.beta_ini = self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """
        Find intial weight value so that the alpha prod values 
        """
        SEARCH_SIZE = 100
        alpha_ini_space = torch.linspace(0., 15., SEARCH_SIZE).unsqueeze(-1)
        beta_ini_space = torch.linspace(-20., 0., SEARCH_SIZE).unsqueeze(-1)

        min_loss = torch.inf

        half_T = self.num_timesteps // 2
        timesteps_center = torch.tensor([half_T - 1, half_T, half_T + 1]).long()
        timesteps_start_end = torch.tensor([0, self.num_timesteps]).long()

        for self.alpha_ini in alpha_ini_space:
            for self.beta_ini in beta_ini_space:
                alpha_prod_t_prev, alpha_prod_t, alpha_prod_t_next = torch.split(self.alpha_cum_prod(timesteps_center), 1)
                alpha_prod_start, alpha_prod_end = torch.split(self.alpha_cum_prod(timesteps_start_end), 1)
                
                d_start = torch.abs(alpha_prod_start - 1.)
                d_end = torch.abs(alpha_prod_end - 0.)

                d_to_center = torch.abs(alpha_prod_t - 0.5)

                slope = (alpha_prod_t_next - alpha_prod_t_prev) / 2.
                slope_center = torch.abs(slope)

                loss = (d_to_center +
                        d_start +
                        d_end +
                        slope_center / 5.)
                
                if loss < min_loss:
                    min_loss = loss
                    alpha_ini = self.alpha_ini
                    beta_ini = self.beta_ini
        return alpha_ini, beta_ini
                                                  
    def forward(self, timesteps):
        eps = 1.0e-6
        t = timesteps / self.num_timesteps
        t_scale = ((1. - t) * torch.clamp(0.9 - self.scale, eps, 1. - eps)) ** torch.clamp(1. + self.shift, 0.4, 2.)
        power = 0.75
        f = lambda x : torch.log((1 + torch.sqrt(1  -  x**2  + eps)) / (x + eps)) # arsech https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions
        #return torch.nn.functional.relu(self.alpha + self.alpha_ini) * f(t_scale) ** power + (self.beta + self.beta_ini) + self.eps * t
        return torch.nn.functional.relu(self.alpha_ini + self.alpha) * f(t_scale) ** power + (self.beta_ini + self.beta) + self.eps * t
