import torch
import torch.nn as nn
from typing import Optional

try:
    from .DDPM import DDPM
except:
    from learning.utils.diffusion.DDPM import DDPM

class CDCD(DDPM):
    def __init__(self,
                 model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(CDCD, self).__init__(model, **kwargs)

    def forward(self,
                sample:torch.Tensor,
                index:torch.Tensor,
                timesteps:Optional[int] = None,
                global_cond:Optional[torch.Tensor] = None,
                criterion:nn.Module = None
                ) -> torch.Tensor:
        """
        Perform forward diffusion process.
        When training return the loss the model output and the
        sample or added noise.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.
            - global_cond: The global conditioning tensor.

        Returns:
            - estimated_noise: Estimated noise.
            - noise: Generated noise.
        """
        if not hasattr(self, "training_shape"):
            training_shape = torch.tensor(sample.shape, device=sample.device).long()
            self.register_buffer("training_shape", training_shape)
        
        if timesteps == None:
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                1,
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
                self.model.eval()
                last_estimate = self.model(noisy_data_samples, timesteps, global_cond, self_conditioning=last_estimate)
                self.model.train()

        logits, model_output = self.model(noisy_data_samples,
                                 timesteps,
                                 global_cond,
                                 self_conditioning=last_estimate,
                                 return_logits = True)

        if self.training and not criterion is None :
            loss = self._compute_loss(logits, index, timesteps, criterion)
            
            return loss
        
        if self.prediction_type == "epsilon":
            alpha_bar_t = torch.take_along_dim(self.noise_scheduler.alphas_cumprod, timesteps)
            model_output = (torch.sqrt(alpha_bar_t) * sample - model_output) / torch.sqrt(1 - alpha_bar_t)

        return model_output
    
    def _compute_loss(self,
                      logits:torch.FloatTensor,
                      index:Optional[torch.FloatTensor],
                      timesteps:torch.IntTensor,
                      criterion:nn.Module,
                      ) -> torch.Tensor:
        
        softplus = torch.nn.functional.softplus

        if self.learn_snr and self.prediction_type == "sample":
            noise_error = softplus((self.noise_scheduler.snr_model.SNR_t_prev_sub_SNR_t(timesteps))) + 1.
            
        elif self.learn_snr and self.prediction_type == "epsilon":
            noise_error = softplus((self.noise_scheduler.snr_model.SNR_t_prev_div_SNR_t(timesteps) - 1.)) + 1.
        
        if self.learn_snr and hasattr(criterion, "reduction"):
            criterion.reduction = "none"
            loss = criterion(logits, index)
            n_dim = len(loss.shape)
            loss = torch.mean(loss, dim=tuple(range(1, n_dim)))
            loss *= noise_error
            loss = torch.mean(loss)

            # log(SNR) penalization, prevent from decreasing too much
            pen_snr_integral = 0.
            pen_snr_middle = 0.

            if self.lambda_snr_integral > 0.:
                T = self.noise_scheduler.snr_model.num_timesteps
                pen_snr_integral = torch.abs(torch.sum(-self.noise_scheduler.snr_model(torch.arange(T))))
                loss += self.lambda_snr_integral * pen_snr_integral

            if self.lambda_snr_middle > 0.:
                T = self.noise_scheduler.snr_model.num_timesteps
                pen_snr_middle = torch.abs(torch.sum(-self.noise_scheduler.snr_model(torch.arange(T))))
                loss += self.lambda_snr_middle * pen_snr_middle

        else:
            loss = criterion(logits, index)
        
        return loss
    
    @torch.no_grad()
    def sample(self,
               num_inference_steps:int=-1,
               size:Optional[torch.Size]=None,
               condition:Optional[torch.Tensor]=None,
               return_intermediate_steps:bool=False
               ) -> torch.Tensor:
        """
        Generate samples using backward diffusion process.

        Args:
            - num_inference_steps: Number of inference steps.
            - size: The size of the generated samples.
            - condition: The conditioning tensor.
            - return_intermediate_steps: Return all denoising steps [STEP, B, SAMPLE_SHAPE]

        Returns:
            - intermediate_generated_samples: Intermediate generated samples.
        """

        assert hasattr(self, "training_shape") or size != None,\
            "Please set attribute 'training_shape' or provide size argument"
        
        device = condition.device
        self.model = self.model.eval()

        # Get sample shape
        if size != None:
            sample_shape = size.tolist()
        elif hasattr(self, "training_shape"):
            sample_shape = self.training_shape.tolist()
        if condition != None:
            sample_shape[0] = condition.shape[0]
        sample_shape = torch.Size(sample_shape)

        # Initial random noise sample
        if num_inference_steps > 0:
            self.noise_scheduler.set_timesteps(num_inference_steps, device)
        else:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps)

        generated_samples = torch.randn(sample_shape, device=device)
        generated_probs = torch.empty(0,)
        generated_pointers = torch.empty(0,)

        # Intermediate diffusion steps
        if return_intermediate_steps:
            intermediate_steps = torch.empty((num_inference_steps + 1, *sample_shape))
            intermediate_probs = []
            intermediate_pointers = []
            
        # Self conditioning
        last_estimate = torch.zeros(sample_shape, device=device)

        # Backward diffusion process
        for i, t in enumerate(self.noise_scheduler.timesteps):

            if return_intermediate_steps:
                intermediate_steps[i] = generated_samples

            timesteps = self.cast_timesteps(generated_samples, t + 1)

            logits, model_output = self.model(generated_samples,
                                      timesteps,
                                      condition,
                                      self_conditioning=last_estimate,
                                      return_logits=True)
            
            if self.self_conditioning:
                last_estimate = model_output

            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=generated_samples
            ).prev_sample

            if return_intermediate_steps:
                intermediate_probs.append(torch.nn.functional.softmax(logits, dim=1).tolist())
                intermediate_pointers.append(self.model.pointers.tolist())

        if return_intermediate_steps:
            intermediate_steps[-1] = generated_samples
            intermediate_probs_tensor = torch.tensor(intermediate_probs)
            intermediate_pointers_tensor = torch.tensor(intermediate_pointers).long()
            return intermediate_steps, intermediate_probs_tensor, intermediate_pointers_tensor

        generated_probs = torch.nn.functional.softmax(logits, dim=-1)
        generated_pointers = self.model.pointers.squeeze(-1)
        
        return generated_samples #, generated_probs, generated_pointers