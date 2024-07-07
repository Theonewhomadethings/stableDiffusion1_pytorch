import torch
import numpy as np

'''
    ### ddpm.py Explained ###

    This script provides a software implementation of the DDPM (Denoising Diffusion Probabilistic Model) sampler, based on the paper "Denoising Diffusion Probabilistic Models" (https://arxiv.org/pdf/2006.11239.pdf). The DDPM sampler is used to define the forward and reverse processes in the latent diffusion model.

    Key components and functionalities of the DDPM sampler include:

    - **Linear Beta Schedule**: The beta schedule defines the variance of noise added at each timestep during the forward diffusion process. This schedule is specified by the `beta_start` and `beta_end` parameters, which are values adopted from the original Stable Diffusion model by the CompVis group (https://github.com/CompVis/stable-diffusion).

    - **Forward and Reverse Processes**: The UNet model predicts the noise in an image, and the DDPM sampler provides the mechanism to iteratively add and remove noise based on the beta schedule. This iterative process enables the generation of high-quality images from noise.

    - **Functions**:
        - `set_inference_timesteps`: Configures the number of timesteps for the inference process, determining how the diffusion process is discretized.
        - `_get_previous_timesteps`: Calculates the previous timestep given the current timestep.
        - `_get_variance`: Computes the variance for the noise to be added during the reverse diffusion process.
        - `set_strength`: Adjusts the strength of the noise added to the input image, controlling how much the output deviates from the input.
        - `step`: Performs a single step in the reverse diffusion process, removing predicted noise from the latent representation.
        - `add_noise`: Adds noise to the original samples based on the beta schedule, simulating the forward diffusion process.

    The DDPM sampler is crucial for the stable diffusion architecture, enabling the transformation of random noise into coherent images guided by learned noise patterns.
'''

class DDPMSampler:

    def __init__(
      self,
      generator: torch.Generator,
      num_training_steps=1000,
      beta_start: float = 0.0085,
      beta_end: float = 0.0120
    ):
        """
        Initializes the DDPMSampler with a given random generator, number of training steps, and beta schedule.
        
        Args:
            generator (torch.Generator): Random number generator for noise sampling.
            num_training_steps (int): Number of steps in the training process.
            beta_start (float): Initial value of the beta schedule.
            beta_end (float): Final value of the beta schedule.
        """
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps=50):
        """
        Configures the number of timesteps for the inference process.
        
        Args:
            num_inference_steps (int): Number of timesteps for the inference process.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_timesteps(self, timestep: int) -> int:
        """
        Calculates the previous timestep given the current timestep.
        
        Args:
            timestep (int): The current timestep.
        
        Returns:
            int: The previous timestep.
        """
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Computes the variance for the noise to be added during the reverse diffusion process.
        
        Args:
            timestep (int): The current timestep.
        
        Returns:
            torch.Tensor: The computed variance.
        """
        prev_t = self._get_previous_timesteps(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sampl
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps*strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
    
    def step(self,
             timestep: int,
             latents: torch.Tensor,
             model_output: torch.Tensor):
        """
        Performs a single step in the reverse diffusion process, removing predicted noise from the latent representation.
        
        Args:
            timestep (int): The current timestep.
            latents (torch.Tensor): The latent representation at the current timestep.
            model_output (torch.Tensor): The model's predicted noise.
        
        Returns:
            torch.Tensor: The latent representation at the previous timestep.
        """
        
        t = timestep
        prev_t = self._get_previous_timesteps(t)

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        #2. Compute predicted original sample from predicited noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)

        # 4. compute ccoefficients for pred_original_sample x_0 and current sampel x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev**(0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t**(0.5) * beta_prod_t_prev / beta_prod_t

        # 5. compute predicted previous sample \mu_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents


        # 6. add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t)**0.5) * noise

        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance 

        return pred_prev_sample


    def add_noise(self,
                  original_samples: torch.FloatTensor,
                  timesteps: torch.IntTensor
                  ) -> torch.FloatTensor:
        """
        Adds noise to the original samples based on the beta schedule, simulating the forward diffusion process.
        
        Args:
            original_samples (torch.FloatTensor): The original samples.
            timesteps (torch.IntTensor): The timesteps at which to add noise.
        
        Returns:
            torch.FloatTensor: The noisy samples.
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype =original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqeeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps])**0.5 # standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device = original_samples.device, dtype = original_samples.dtype)
                    # According to equation 4 of DDPM Paper: mean + standarddeviation*noise
        noisy_samples = sqrt_alpha_prod*original_samples + sqrt_one_minus_alpha_prod*noise 


        return noisy_samples
