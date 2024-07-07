from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

def preload_models_from_standard_weights(ckpt_path, device):
    """
    Preloads models with weights from a specified checkpoint file.

    This function loads pre-trained weights from a checkpoint file into 
    the respective models (VAE Encoder, VAE Decoder, Diffusion model, and CLIP model)
    and moves them to the specified device (CPU or GPU).

    Args:
        ckpt_path (str): Path to the checkpoint file containing the pre-trained weights.
        device (torch.device): The device to which the models should be moved.

    Returns:
        dict: A dictionary containing the loaded models with keys "clip", "encoder", "decoder", and "diffusion".
    """

    # Load the state dictionary from the checkpoint file
    state_dict = model_coverter.load_from_standard_weights(ckpt_path, device)

    # Initialize the VAE Encoder and load its weights
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    # Initialize the VAE Decoder and load its weights
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    # Initialize the Diffusion model and load its weights
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    # Initialize the CLIP model and load its weights
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    # Return a dictionary of the loaded models
    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }
