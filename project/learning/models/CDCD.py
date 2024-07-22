import torch
import torch.nn as nn
import time
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Mapping, Optional, Union
from .DDPM import DDPM

DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 2e-2
DEFAULT_BETASCHEDULE = 'squaredcos_cap_v2'
DEFAULT_TIMESTEPS = 1000
DEFAULT_CLIPSAMPLE = False
DEFAULT_TRAINING_SHAPE = [0,0,0]
DEFAULT_PREDICTION_TYPE = "epsilon" # "sample" or "epsilon"
DEFAULT_SELF_CONDITIONING = False

class CDCD(DDPM):
    def __init__(self,
                 eps_model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - eps_model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(CDCD, self).__init__(eps_model, **kwargs)

    def forward(self,
                sample:torch.Tensor,
                timesteps:Optional[int] = None,
                global_cond:Optional[torch.Tensor] = None):
        """
        Perform forward diffusion process.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.
            - global_cond: The global conditioning tensor.

        Returns:
            - estimated_noise: Estimated noise.
            - noise: Generated noise.
        """
        if (self.training_shape == torch.zeros((3,), device=sample.device)).all():
            self.training_shape = torch.tensor(sample.shape, device=sample.device).long()
            
        if timesteps == None:
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (sample.shape[0],),
                device=sample.device
            ).long()

        timesteps = self.cast_timesteps(sample, timesteps)

        # Forward diffusion process
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_data_samples = self.noise_scheduler.add_noise(sample, noise, timesteps)

        # Diffusion process
        last_estimate = torch.zeros_like(noise, device=sample.device)
        if torch.rand(1) < int(self.self_conditioning) / 2:
            with torch.no_grad():
                self.eps_model.eval()
                last_estimate = self.eps_model(noisy_data_samples, timesteps, global_cond, self_conditioning=last_estimate, return_logits=False)
                self.eps_model.train()

        estimated_logits, sample = self.eps_model(noisy_data_samples, timesteps, global_cond, self_conditioning=last_estimate, return_logits=True)

        if self.prediction_type == "epsilon":
            return estimated_logits, noise
        
        elif self.prediction_type == "sample":
            return estimated_logits, sample
    
    @torch.no_grad()
    def sample(self,
               size:Optional[torch.Size]=None,
               num_inference_steps:int=-1,
               condition:Optional[torch.Tensor]=None,
               return_intermediate_steps:bool=False
               ):
        """
        Generate samples using backward diffusion process.

        Args:
            - size: The size of the generated samples.
            - num_inference_steps: Number of inference steps.
            - condition: The conditioning tensor.
            - return_intermediate_steps: Return all denoising steps [STEP, B, SAMPLE_SHAPE]

        Returns:
            - intermediate_generated_samples: Intermediate generated samples.
        """

        assert (hasattr(self, "training_shape") or size != None),\
            "Please set attribute sample_shape or provide size argument"
        
        device = "cpu" if not(torch.cuda.is_available()) else "cuda"
        device = condition.device if condition != None else device

        # Get sample shape
        if size != None:
            sample_shape = size.tolist()
        elif hasattr(self, "training_shape"):
            sample_shape = self.training_shape.tolist()
        if condition != None:
            sample_shape[0] = condition.shape[0]
        sample_shape = torch.Size(sample_shape)

        # Initial random noise sample
        seed = time.time_ns() // 100
        torch.random.manual_seed(seed)

        generated_samples = torch.randn(sample_shape, device=device)
        if hasattr(self, "num_inference_steps"):
            num_inference_steps = self.num_inference_steps
        if num_inference_steps < 0:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps - 1
        self.noise_scheduler.set_timesteps(num_inference_steps, device)

        # Self conditioning
        last_estimate = torch.zeros(sample_shape, device=device)

        # Intermediate diffusion steps
        intermediate_steps = torch.empty((num_inference_steps + 1, *sample_shape))

        # Backward diffusion process
        for i, t in enumerate(self.noise_scheduler.timesteps):

            if return_intermediate_steps:
                intermediate_steps[i] = generated_samples

            timesteps = self.cast_timesteps(generated_samples, t)
            noise_pred = self.eps_model(generated_samples, 
                                        timesteps,
                                        condition,
                                        self_conditioning=last_estimate,
                                        return_logits=False)

            if self.self_conditioning:
                last_estimate = noise_pred
            
            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=generated_samples
            )[0]

        if return_intermediate_steps:
            intermediate_steps[-1] = generated_samples
            return intermediate_steps
        
        return generated_samples
    
    @torch.no_grad()
    def select(self,
               size:Optional[torch.Size]=None,
               num_inference_steps:int=-1,
               condition:Optional[torch.Tensor]=None,
               ):
        self.sample(size, num_inference_steps, condition)
        return self.eps_model.selected