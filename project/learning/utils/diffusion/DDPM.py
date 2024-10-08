import torch
import torch.nn as nn
from .noise_scheduler import NoiseScheduler
from typing import Any, Mapping, Optional


class DDPM(nn.Module):

    DEFAULT_SELF_CONDITIONING = False
    DEFAULT_BETA_START = 1e-4
    DEFAULT_BETA_END = 0.02
    DEFAULT_BETASCHEDULE = 'squaredcos_cap_v2'
    DEFAULT_TIMESTEPS = 1000
    DEFAULT_CLIPSAMPLE = True
    DEFAULT_PREDICTION_TYPE = "epsilon" # "sample" or "epsilon"
    DEFAULT_LEARN_SNR = False
    DEFAULT_COEF_SNR_INTEGRAL_PEN = 0.001
    DEFAULT_COEF_SNR_HALF_PEN = 0.
    DEFAULT_SCALE_LOSS_SNR = 0.

    def __init__(self,
                 model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(DDPM, self).__init__()
        self.model = model

        self.optional_parameters = {
            "beta_start" : DDPM.DEFAULT_BETA_START,
            "beta_end" : DDPM.DEFAULT_BETA_END,
            "beta_schedule" : DDPM.DEFAULT_BETASCHEDULE,
            "timesteps" : DDPM.DEFAULT_TIMESTEPS,
            "clip_sample" : DDPM.DEFAULT_CLIPSAMPLE,
            "self_conditioning" : DDPM.DEFAULT_SELF_CONDITIONING,
            "prediction_type" : DDPM.DEFAULT_PREDICTION_TYPE,
            "learn_snr" : DDPM.DEFAULT_LEARN_SNR,
            "lambda_snr_integral" : DDPM.DEFAULT_COEF_SNR_INTEGRAL_PEN,
            "lambda_snr_middle" : DDPM.DEFAULT_COEF_SNR_HALF_PEN,
            "gamma_scale_loss_snr" : DDPM.DEFAULT_SCALE_LOSS_SNR,
        }

        self._detach_snr_gradient = False
        self.extra_states = {}
        self._set_parameters(**kwargs)
        self._set_noise_scheduler()

    def _set_parameters(self, **kwargs):
        """
        Set optional parameters as attributes or buffers.
        """
        # Update optional parameters values based on kwargs
        for key, value in kwargs.items():
            if key in self.optional_parameters:
                self.optional_parameters[key] = value

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in self.optional_parameters.items():
            try:
                v = torch.tensor(v)
                self.register_buffer(k, v)
            except:
                setattr(self, k, v)
                self.extra_states[k] = v
            

    def _set_noise_scheduler(self):
        """
        Initialize and set the noise scheduler.
        """
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=self.timesteps.item(),
            learn_snr=self.learn_snr,
            beta_start=self.beta_start.item(),
            beta_end=self.beta_end.item(),
            beta_schedule=self.beta_schedule, # big impact on performance
            clip_sample=self.clip_sample.item(), # clip output to [-1,1] to improve stability
            prediction_type=self.prediction_type # our network predicts noise (instead of denoised action)
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.register_buffer("training_shape", state_dict["training_shape"])
        
        super().load_state_dict(state_dict, False, assign)
        
        if self.learn_snr:
            self._set_noise_scheduler()
            self.noise_scheduler.snr_model.load_state_dict(state_dict)


    def set_extra_state(self, state: Any):
        self.extra_states.update(state)
        for k, v in self.extra_states.items(): setattr(self, k, v)
    
    def get_extra_state(self) -> Any:
        return self.extra_states

    def cast_timesteps(self,
                       sample:torch.Tensor,
                       timesteps:Optional[int]):
        """
        Cast and broadcast timesteps.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.

        Returns:
            - The cast and broadcasted timesteps.
        """
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        return timesteps
        
    def forward(self,
                sample:torch.Tensor,
                timesteps:Optional[int] = None,
                condition:Optional[torch.Tensor] = None,
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
                last_estimate = self.model(noisy_data_samples, timesteps, condition, self_conditioning=last_estimate)
                self.model.train()

        model_output = self.model(noisy_data_samples,
                                 timesteps,
                                 condition,)
                                 #self_conditioning=last_estimate)

        if self.training and not criterion is None :
            loss = self._compute_loss(model_output, sample, noise, timesteps, criterion)
            return loss
        
        return model_output
    
    def _compute_loss(self,
                      model_output:torch.FloatTensor,
                      sample:Optional[torch.FloatTensor],
                      noise:Optional[torch.FloatTensor],
                      timesteps:torch.IntTensor,
                      criterion:nn.Module,
                      ) -> torch.Tensor:
        
        target = noise if self.prediction_type == "epsilon" else sample
        softplus = torch.nn.functional.softplus

        if self.learn_snr and self.prediction_type == "sample":
            noise_error = softplus((self.noise_scheduler.snr_model.SNR_t_prev_sub_SNR_t(timesteps))) + 1.
            
        elif self.learn_snr and self.prediction_type == "epsilon":
            noise_error = softplus((self.noise_scheduler.snr_model.SNR_t_prev_div_SNR_t(timesteps) - 1.)) + 1.
        
        if self.learn_snr and hasattr(criterion, "reduction"):
            criterion.reduction = "none"
            loss = criterion(model_output, target)
            n_dim = len(loss.shape)
            loss = torch.mean(loss, dim=tuple(range(1, n_dim)))
            loss *= noise_error
            loss = torch.mean(loss)

            T = self.noise_scheduler.snr_model.num_timesteps

            # log(SNR) penalization, prevent from decrprediction_type

            if self.lambda_snr_integral > 0.:
                pen_snr_integral = torch.abs(torch.sum(-self.noise_scheduler.snr_model(torch.arange(T))))
            
            if self.lambda_snr_middle > 0.:
                pen_snr_middle = torch.abs(torch.sum(-self.noise_scheduler.snr_model(torch.arange(T))))
            
            loss += self.lambda_snr_integral * pen_snr_integral + self.lambda_snr_middle * pen_snr_middle
        else:
            criterion.reduction = "mean"
            loss = criterion(model_output, target)

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
        generated_samples = torch.randn(sample_shape, device=device)
        if num_inference_steps > 0:
            self.noise_scheduler.set_timesteps(num_inference_steps, device)
        else:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            self.noise_scheduler.set_timesteps(num_inference_steps)
        
        # Self conditioning
        last_estimate = torch.zeros(sample_shape, device=device)

        # Intermediate diffusion steps
        intermediate_steps = torch.empty((num_inference_steps + 1, *sample_shape))

        # Backward diffusion process
        for i, t in enumerate(self.noise_scheduler.timesteps):

            if return_intermediate_steps:
                intermediate_steps[i] = generated_samples

            timesteps = self.cast_timesteps(generated_samples, t)

            model_output = self.model(generated_samples, timesteps, condition, self_conditioning=last_estimate)
            
            if self.self_conditioning:
                last_estimate = model_output

            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=generated_samples
            ).prev_sample

        if return_intermediate_steps:
            intermediate_steps[-1] = generated_samples
            return intermediate_steps
        
        return generated_samples