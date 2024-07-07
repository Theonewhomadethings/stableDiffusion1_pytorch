import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

'''
    ### diffusion.py Explained ###

    This script implements the core components of the UNet model within the stable diffusion architecture, including the time embedding, UNet residual blocks, and UNet attention blocks. (UNet first paper, https://arxiv.org/abs/1505.04597), (UNet attention first paper, https://arxiv.org/abs/1804.03999)

    The UNet model plays a crucial role in the stable diffusion process by predicting the noise present in a given image and determining how to remove it. This is essential for reconstructing the image from its noisy version through a reverse diffusion process, while the forward process is governed by a predefined schedule, such as DDPM (Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239).

    The UNet model requires not only the noisy image at a specific timestep but also the textual prompt that guides the generation process. This integration of image and text information is facilitated by the CrossAttention mechanism, which computes attention between two sequences, combining image and text modalities. Cross attention ensures that the model understands the relationship between the prompt and the noisy image, enabling accurate reconstruction of the desired output.

    Key components in this script include:
    - TimeEmbedding: Encodes the timestep information.
    - UNET_ResidualBlock: Processes the image features and integrates time embeddings.
    - UNET_AttentionBlock: Applies both self-attention and cross-attention mechanisms.
    - SwitchSequential: Manages the sequence of operations in the UNet.
    - UNET: Constructs the entire UNet model.
    - UNET_OutputLayer: Final layer that produces the output image.
    - Diffusion: Combines the UNet and other components to perform the denoising process.
'''


class TimeEmbedding(nn.Module):
    '''
    Encodes the timestep information into a higher dimensional space using two linear layers.

    Attributes:
        linear_1 (nn.Linear): First linear layer to project the input.
        linear_2 (nn.Linear): Second linear layer to further transform the projection.
    '''
    def __init__(self, n_embd):
        '''
        Initializes the TimeEmbedding with the specified embedding size.

        Args:
            n_embd (int): Dimensionality of the input embedding.
        '''
        super().__init__()

        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)

    def forward(self, x):
        '''
        Forward pass to transform the timestep embedding.

        Args:
            x (torch.Tensor): Input tensor with shape (1, n_embd).

        Returns:
            torch.Tensor: Transformed tensor with shape (1, 4 * n_embd).
        '''

        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        # (1, 1280) -> (1, 1280)
        x = F.silu(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x
    
class UNET_ResidualBlock(nn.Module):
    '''
    Processes image features and integrates time embeddings using residual connections.

    Attributes:
        groupnorm_feature (nn.GroupNorm): Group normalization for feature normalization.
        conv_feature (nn.Conv2d): Convolutional layer for feature processing.
        linear_time (nn.Linear): Linear layer to transform the time embedding.
        groupnorm_merged (nn.GroupNorm): Group normalization after merging features and time.
        conv_merged (nn.Conv2d): Convolutional layer after merging features and time.
        residual_layer (nn.Module): Identity or convolution layer for residual connection.
    '''
    def __init__(self, in_channels, out_channels, n_time=1280):
        '''
        Initializes the UNET_ResidualBlock with specified input and output channels.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_time (int): Dimensionality of the time embedding.
        '''
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    
    def forward(self, feature, time):
        '''
        Forward pass to process features and integrate time embeddings.

        Args:
            feature (torch.Tensor): Feature tensor with shape (batch_size, in_channels, height, width).
            time (torch.Tensor): Time embedding tensor with shape (1, n_time).

        Returns:
            torch.Tensor: Output tensor with the same shape as the feature input.
        '''
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

    
class UNET_AttentionBlock(nn.Module):
    '''
    Applies both self-attention and cross-attention mechanisms with residual connections.

    Attributes:
        groupnorm (nn.GroupNorm): Group normalization for feature normalization.
        conv_input (nn.Conv2d): Convolutional layer for initial feature processing.
        layernorm_1 (nn.LayerNorm): Layer normalization before self-attention.
        attention_1 (SelfAttention): Self-attention mechanism.
        layernorm_2 (nn.LayerNorm): Layer normalization before cross-attention.
        attention_2 (CrossAttention): Cross-attention mechanism.
        layernorm_3 (nn.LayerNorm): Layer normalization before feed-forward network.
        linear_geglu_1 (nn.Linear): First linear layer with GeGLU activation.
        linear_geglu_2 (nn.Linear): Second linear layer in the feed-forward network.
        conv_output (nn.Conv2d): Convolutional layer for final feature processing.
    '''
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        '''
        Initializes the UNET_AttentionBlock with specified number of heads, embedding dimension, and context dimension.

        Args:
            n_head (int): Number of attention heads.
            n_embd (int): Dimensionality of the embeddings.
            d_context (int): Dimensionality of the context.
        '''
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        '''
        Forward pass through the attention block.

        Args:
            x (torch.Tensor): Feature tensor with shape (batch_size, features, height, width).
            context (torch.Tensor): Context tensor with shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as the feature input.
        '''
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    '''
    Upsamples the input tensor by a factor of 2 using nearest neighbor interpolation followed by a convolution.

    Attributes:
        conv (nn.Conv2d): Convolutional layer to refine the upsampled features.
    '''
    def __init__(self, channels):
        '''
        Initializes the Upsample layer with specified number of channels.

        Args:
            channels (int): Number of input and output channels.
        '''
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        '''
        Forward pass to upsample the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Upsampled tensor with shape (batch_size, channels, height*2, width*2).
        '''
        x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    '''
    Custom sequential layer that handles different types of layers, including attention and residual blocks.

    This allows dynamic handling of inputs based on the type of the layer (e.g., applying attention with context).
    '''
    def forward(self, x, context, time):
        '''
        Forward pass through the layers, applying context and time where necessary.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor): Context tensor for cross-attention.
            time (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the sequential layers.
        '''
        for layer in self:
            if (isinstance(layer, UNET_AttentionBlock)):
                x = layer(x, context)
            elif (isinstance(layer, UNET_ResidualBlock)):
                x = layer(x, time)
            else:
                x = layer(x)
            return x
    
class UNET(nn.Module):
    '''
    Constructs the entire UNet model, including encoding, bottleneck, and decoding stages.

    The UNet architecture in this implementation consists of:
    - 12 encoding layers that progressively reduce the spatial dimensions and increase the number of feature channels.
    - 3 bottleneck layers that process the most compressed representation.
    - 12 decoding layers that progressively increase the spatial dimensions and reduce the number of feature channels, ultimately restoring the original resolution.

    Encoder:
        Input: (Batch_Size, 4, Height / 8, Width / 8)
        Output: (Batch_Size, 1280, Height / 64, Width / 64)

    Bottleneck:
        Input: (Batch_Size, 1280, Height / 64, Width / 64)
        Output: (Batch_Size, 1280, Height / 64, Width / 64)

    Decoder:
        Input: (Batch_Size, 1280, Height / 64, Width / 64)
        Output: (Batch_Size, 4, Height / 8, Width / 8)

    Attributes:
        encoders (nn.ModuleList): List of encoding layers that reduce spatial dimensions and increase feature channels.
        bottleneck (SwitchSequential): Bottleneck layers that process the most compressed representation of the input.
        decoders (nn.ModuleList): List of decoding layers that increase spatial dimensions and reduce feature channels to restore the original resolution.
    '''
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),          

        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UNET_OutputLayer(nn.Module):
    '''
    Final output layer for the UNet model, applying normalization and a convolution.

    Attributes:
        groupnorm (nn.GroupNorm): Group normalization for feature normalization.
        conv (nn.Conv2d): Convolutional layer to produce the final output.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size =3, padding = 1)


    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x

class Diffusion(nn.Module):
    '''
    Diffusion model that combines the UNet, time embedding, and final output layer.

    Attributes:
        time_embedding (TimeEmbedding): Encodes the timestep information.
        unet (UNET): UNet model for processing the noisy image and context.
        final (UNET_OutputLayer): Produces the final denoised image.
    '''
    def __init__(self):
        super().__init__()
                              #TimeEmbedding(size)
        self.time_embedding = TimeEmbedding(320)

        self.unet = UNET()

        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent (batch_size, 4, Height/8, Width/8)
        # context (batch_size,sequence_length,Dimension)
        # time (1, 320)
        
        #(1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        #(batch_size, 4, Height/8, Width/8) -> (batch_size, 320, Height/8, Width/8)
        output = self.unet(latent, context, time)

        #(batch_size, 320, Height/8, Width/8) -> (batch_size, 4, Height/8, Width/8)
        output = self.final(output)

        return output
