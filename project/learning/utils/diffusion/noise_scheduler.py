from typing import Mapping
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from typing import Any

from .snr_model import SNRModelSigmoid, SNRModelArsech, SNRModelCumulative

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
        beta_schedule = kwargs.pop("beta_schedule", NoiseScheduler.AVAILABLE_BETA_SCHEDULE[-2])

        if beta_schedule == "sigmoid":
            self.snr_model = SNRModelSigmoid(num_train_timesteps)
        elif beta_schedule == "arsech":
            self.snr_model = SNRModelArsech(num_train_timesteps)
        elif beta_schedule == "cumulative":
            self.snr_model = SNRModelCumulative(num_train_timesteps)
        else: # Default
            self.snr_model = None
            self.learn_snr = False

        if (not beta_schedule in NoiseScheduler.AVAILABLE_BETA_SCHEDULE):
            beta_schedule = "squaredcos_cap_v2"
            
        DDPMScheduler.__init__(self,
                               num_train_timesteps=num_train_timesteps,
                               beta_schedule = beta_schedule,
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
    