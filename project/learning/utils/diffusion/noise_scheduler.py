import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from typing import Any

from .snr_model import SNRModelSigmoid, SNRModelArsech, SNRModelCumulative

def get_coef(t, alpha_prod):
    alpha_prod_t = alpha_prod[t]
    alpha_prod_t_prev = alpha_prod[t-1] if t > 0 else 1.
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    
    return pred_original_sample_coeff.item(), current_sample_coeff.item()

def get_variance(t, alpha_prod):
    alpha_prod_t = alpha_prod[t]
    alpha_prod_t_prev = alpha_prod[t-1] if t > 0 else 1.
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    sigma_square_t = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

    return sigma_square_t.item()

class NoiseScheduler(nn.Module, DDPMScheduler):
    AVAILABLE_BETA_SCHEDULE = [
        "linear",
        "scaled_linear",
        "squaredcos_cap_v2",
        "sigmoid"
    ]
    """
    Noise scheduler with learnable noise values.
    """
    def __init__(self,
                 num_train_timesteps:int,
                 learn_snr:bool=False,
                 **kwargs) -> None:
        nn.Module.__init__(self)
        
        self.learn_snr = learn_snr
        self.beta_schedule = kwargs.pop("beta_schedule", NoiseScheduler.AVAILABLE_BETA_SCHEDULE[-2])

        if self.beta_schedule == "sigmoid":
            self.snr_model = SNRModelSigmoid(num_train_timesteps)
        elif self.beta_schedule == "arsech":
            self.snr_model = SNRModelArsech(num_train_timesteps)
        elif self.beta_schedule == "cumulative":
            self.snr_model = SNRModelCumulative(num_train_timesteps)
        else: # Default
            self.snr_model = None
            self.learn_snr = False

        if (not self.beta_schedule in NoiseScheduler.AVAILABLE_BETA_SCHEDULE):
            self.beta_schedule = "squaredcos_cap_v2"
            
        DDPMScheduler.__init__(self,
                               num_train_timesteps=num_train_timesteps,
                               beta_schedule = self.beta_schedule,
                               **kwargs)
        
        self._update_alphas()

    def _update_alphas(self):
        if self.learn_snr:
            device = next(self.snr_model.parameters()).device
            self.alphas_cumprod = self.snr_model.alpha_cum_prod(torch.arange(self.config.num_train_timesteps, device=device))

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        self._update_alphas()
        return super().add_noise(original_samples, noise, timesteps)

    @torch.no_grad()
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, generator=None, return_dict: bool = True) -> DDPMSchedulerOutput:
        self._update_alphas()
        return super().step(model_output, timestep, sample, generator, return_dict)
    
    def snr(self, timesteps):
        snr_schedule = self.alphas_cumprod / (1 - self.alphas_cumprod)
        snr_t = torch.take_along_dim(snr_schedule, timesteps, dim=0)
        return snr_t
    
    def plot_diffusion_schedule(self):
        T = len(self.alphas_cumprod)
        # Alpha cumulative product
        alpha_prod_scheduler = self.alphas_cumprod
        # Signal to Noise ratio
        snr_scheduler = alpha_prod_scheduler / (1 - alpha_prod_scheduler)
        log_snr_scheduler = torch.log(snr_scheduler)
        # SNR(t) - SNR(t-1)
        snr_diff_scheduler = snr_scheduler[1:] - snr_scheduler[:-1]
        # SNR(t) / SNR(t-1)
        snr_div_scheduler = snr_scheduler[1:] / snr_scheduler[:-1]
        # Original and Previous coefficients
        coef_original_scheduler = [get_coef(t, alpha_prod_scheduler)[0] for t in range(T)]
        coef_previous_scheduler = [get_coef(t, alpha_prod_scheduler)[1] for t in range(T)]
        variance_scheduler = [get_variance(t, alpha_prod_scheduler) for t in range(T)]

        # Create the softplus lambda function
        f = lambda x: torch.log(x + 1.) + 1.

        # Create subplots
        fig, axs = plt.subplots(2, 4, figsize=(12, 5))  # 4 rows, 2 columns of plots
        fig.subplots_adjust(hspace=0.4)  # Space between plots

        # Alpha cumulative product
        axs[0, 0].plot(alpha_prod_scheduler)
        axs[0, 0].set_title("Alpha cumulative product (t)")

        # Log SNR
        axs[1, 0].plot(log_snr_scheduler)
        axs[1, 0].set_title("log(SNR(t))")

        # Coeff original
        axs[0, 1].plot(coef_original_scheduler)
        axs[0, 1].set_title("Coeff original (t)")

        # Coeff previous
        axs[1, 1].plot(coef_previous_scheduler)
        axs[1, 1].set_title("Coeff previous (t)")

        # Variance
        axs[0, 2].plot(variance_scheduler)
        axs[0, 2].set_title("Variance schedule (t)")

        # softplus(SNR(t-1) - SNR(t)) + 1
        axs[1, 2].plot(f(snr_diff_scheduler) + 1.)
        axs[1, 2].set_title("softplus(SNR(t-1) - SNR(t)) + 1")

        # softplus(SNR(t-1) / SNR(t)) + 1
        axs[0, 3].plot(f(snr_div_scheduler))
        axs[0, 3].set_title("softplus(SNR(t-1) / SNR(t)) + 1")

        # Remove the empty subplot (bottom-right)
        fig.suptitle(f"Schedule {self.beta_schedule}, beta start: {self.betas[0]:.3f}, beta end: {self.betas[-1]:.3f}")
        fig.delaxes(axs[1, 3])

        # Show the full figure
        plt.show()