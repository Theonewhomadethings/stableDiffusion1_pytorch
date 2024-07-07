import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512 
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8 

"""
    ### pipeline.py Explained ###

    This script is a core component of the Stable Diffusion implementation. It orchestrates the entire process of generating images from text prompts or input images using diffusion models. 
    The script leverages several models and techniques, including CLIP for text embeddings, a UNet-based diffusion model, and DDPMSampler for the diffusion process. 
    Here is a breakdown of the functionalities:

    - **generate Function**:
        This is the main function that generates images based on various inputs and parameters. It performs the following steps:

        1. **Input Validation**: 
            Ensures that the `strength` parameter is within the valid range.

        2. **Random Generator Initialization**: 
            Initializes a random number generator with a given seed for reproducibility.

        3. **Model Setup**: 
            Loads the CLIP model and moves it to the specified device (CPU or GPU). Depending on the classifier-free guidance (`do_cfg`), it processes the text prompt and optionally an unconditional prompt to create text embeddings using CLIP.

        4. **Sampler Initialization**: 
            Initializes the DDPMSampler and sets the number of inference steps.

        5. **Latent Initialization**: 
            If an input image is provided, it is encoded into a latent space representation using the encoder model. Noise is added to this latent representation based on the `strength` parameter. If no input image is provided, random latents are initialized.

        6. **Diffusion Process**: 
            Moves the diffusion model to the specified device. Iterates over the timesteps to progressively denoise the latent representation. Uses the diffusion model to predict and remove noise at each step. If classifier-free guidance is enabled, adjusts the model output accordingly.

        7. **Decoding**: 
            Moves the decoder model to the specified device and decodes the final latent representation back into an image. The image tensor is then rescaled from latent space to pixel space and converted to a NumPy array for output.

    - **Helper Functions**:
        - `rescale`: Rescales tensor values from an old range to a new range, with optional clamping.
        - `get_time_embedding`: Generates a time embedding tensor for a given timestep.

    The generate function is highly configurable, allowing for various modes of operation such as text-to-image generation, image-to-image translation, and the use of classifier-free guidance. It integrates multiple models and components seamlessly to produce high-quality images based on the given prompts and input parameters.

    This script is designed to be used as part of a larger pipeline for generating images using Stable Diffusion.
"""


def generate(
        prompt,
        uncond_prompt=None, # negative prompt or empty strinng
        input_image=None,  # image to image 
        strength=0.8, # how much noise is added to latent input image when image to image
        do_cfg=True, # if you want classifier free guidance 
        cfg_scale=7.5,  # weight of how much you wnat the model to pay attention to the prompt value goes from 1 -> 14 
        sampler_name="ddpm",
        n_inference_steps=50, # timesteps for scheduler 1000 = 50 * 20 so 50 intervals of 20 1000, 980, 960 .... 0
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # (Batch_Size, Seq_Len)
            cond_tokens=torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)

            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids

            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
             # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids

            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        
        to_idle(clip)


        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH,HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
             # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timesteps).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale*(output_cond - output_uncond) + output_uncond

            # remove noise predicted by UNET
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)    

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(decoder)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)


        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        images= images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
   
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
   
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
