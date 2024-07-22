import torch
import torch.nn as nn
import time
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Mapping, Optional, Union

DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02
DEFAULT_BETASCHEDULE = 'squaredcos_cap_v2'
DEFAULT_TIMESTEPS = 1000
DEFAULT_CLIPSAMPLE = False
DEFAULT_TRAINING_SHAPE = [0,0,0]
DEFAULT_LEARN_SNR = False
DEFAULT_PREDICTION_TYPE = "epsilon" # "sample" or "epsilon"
DEFAULT_SELF_CONDITIONING = False

class DDPM(nn.Module):
    def __init__(self,
                 eps_model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - eps_model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(DDPM, self).__init__()
        self.eps_model = eps_model

        self.optional_parameters = {
            **{
            "beta_start" : DEFAULT_BETA_START,
            "beta_end" : DEFAULT_BETA_END,
            "beta_schedule" : DEFAULT_BETASCHEDULE,
            "timesteps" : DEFAULT_TIMESTEPS,
            "clip_sample" : DEFAULT_CLIPSAMPLE,
            "training_shape" : DEFAULT_TRAINING_SHAPE,
            "learn_snr" : DEFAULT_LEARN_SNR,
            "prediction_type" : DEFAULT_PREDICTION_TYPE,
            "self_conditioning" : DEFAULT_SELF_CONDITIONING,
            },
            **kwargs # kwargs overrides optional if provided
        }
        self._set_parameters()
        self._set_noise_scheduler()

    def _set_parameters(self):
        """
        Set optional parameters as attributes or buffers.
        """
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in self.optional_parameters.items():
            try:
                v = torch.tensor(v)
            except:
                setattr(self, k, v)
                continue
            self.register_buffer(k, v)

    def _set_noise_scheduler(self):
        """
        Initialize and set the noise scheduler.
        """
        if not self.learn_snr:
            self.noise_scheduler = DDPMScheduler(
                beta_start=self.beta_start.item(),
                beta_end=self.beta_end.item(),
                beta_schedule=self.beta_schedule, # big impact on performance
                num_train_timesteps=self.timesteps.item(),            
                clip_sample=self.clip_sample.item(), # clip output to [-1,1] to improve stability
                prediction_type=self.prediction_type # our network predicts noise (instead of denoised action)
            )
        else:
            self.snr_model = SNRModel(self.timesteps.item())
            self.mean_snr = torch.ones(1)
            self.noise_scheduler = LearnableNoiseScheduler(self.snr_model,
                                                           timesteps=self.timesteps.item(),
                                                           prediction_type=self.prediction_type) 

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict, assign)
        self._set_noise_scheduler()
        if self.learn_snr:
            state_snr = {k.replace("snr_model.", ""): v for k, v in state_dict.items() if "snr_model" in k}
            self.snr_model.load_state_dict(state_snr)

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
                2,
                self.noise_scheduler.config.num_train_timesteps,
                (sample.shape[0],),
                device=sample.device
            ).long()

        timesteps = self.cast_timesteps(sample, timesteps)

        # Forward diffusion process
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_data_samples = self.noise_scheduler.add_noise(sample, noise, timesteps)
        estimated_noise = self.eps_model(noisy_data_samples, timesteps, global_cond)

        if self.learn_snr:
            self.mean_snr = 0.5 * (self.snr_model.SNR(timesteps-1) - self.snr_model.SNR(timesteps))

        if self.prediction_type == "epsilon":
            return estimated_noise, noise
    
        if self.prediction_type == "sample":
            return estimated_noise, sample
    
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
            noise_pred = self.eps_model(generated_samples, timesteps, condition)
            
            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=generated_samples,
                return_dict=False,
            )[0]

        if return_intermediate_steps:
            intermediate_steps[-1] = generated_samples
            return intermediate_steps
        
        return generated_samples

class SNRModel(nn.Module):
    """
    Monotically increasing NN that represent SNR ratio.
    Output is between 0 and 1.
    """
    def __init__(self, num_timesteps) -> None:
        super(SNRModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.slope = torch.nn.Parameter(torch.ones((1,)) * 4. , requires_grad=True)
        self.power = torch.nn.Parameter(torch.zeros((1,)) , requires_grad=False)
        self.inf = torch.full((1,), torch.inf)
        self.W = torch.nn.Parameter(torch.zeros((self.num_timesteps)), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        """
        Find intial weight value so that alpha_prod(T/2) = 0.5
        """
        w_ini_space = torch.linspace(0.01, 1., 100)
        d_to_center = torch.ones_like(w_ini_space)

        for i, self.w_ini in enumerate(w_ini_space):
            half_T = torch.tensor(self.num_timesteps // 2).unsqueeze(0).long()
            d_to_center[i] = torch.abs(self.alpha_prod(half_T) - 0.5)

        self.w_ini = w_ini_space[torch.argmin(d_to_center)]

    def alpha_prod(self, timesteps):
        return torch.nn.functional.sigmoid(-(1. + self.power)*self.forward(timesteps))
    
    def SNR(self, timesteps):
        return torch.exp(-(1. + self.power) * self.forward(timesteps))

    def forward(self, timesteps):
        W = torch.cumsum(torch.nn.functional.relu(self.W + self.w_ini), dim=0)
        W -= self.slope
        W = torch.cat((-self.inf, W), dim=-1)
        return torch.take_along_dim(W, timesteps, dim=-1)

class LearnableNoiseScheduler():
    def __init__(self,
                 snr_model:nn.Module,
                 timesteps:int=DEFAULT_TIMESTEPS,
                 prediction_type:str="epsilon",
                 **kwargs) -> None:
        self.snr_model = snr_model
        self.num_train_timesteps = timesteps
        self.num_inference_steps = None
        self.timesteps = self.set_timesteps(timesteps)
        self.prediction_type = prediction_type
    
        class Config:
            num_train_timesteps = self.num_train_timesteps
            num_inference_steps = None
            prediction_type = self.prediction_type
    
        self.config = Config

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        timesteps = (
            np.linspace(1, self.num_train_timesteps, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def previous_timestep(self, timestep):

        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        prev_t = timestep - self.num_train_timesteps // num_inference_steps

        return prev_t

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:

        alphas_cumprod = self.snr_model.alpha_prod(timesteps)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        **kwargs,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.snr_model.alpha_prod(t.long().expand(1))
        alpha_prod_t_prev = self.snr_model.alpha_prod(prev_t.long().expand(1))
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0.
        if t > 0:
            variance_noise = torch.randn_like(model_output, device=model_output.device, dtype=model_output.dtype)
            sigma_square_t = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            # we always take the log of variance, so clamp it to ensure it's not 0
            variance =  sigma_square_t ** 0.5 * variance_noise

        pred_prev_sample += variance

        return (pred_prev_sample, pred_original_sample_coeff, current_sample_coeff)