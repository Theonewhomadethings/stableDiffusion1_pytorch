import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

'''
    ### encoder.py Explained ### 

This script implements the encoder component of the Variational Autoencoder (VAE), which is responsible for compressing input images into a latent space representation. The encoder processes the input image to generate a mean and a variance, from which samples are drawn to obtain the latent representation.

The primary purpose of this script is to encode an image (for Image-to-Image tasks) or random noise (for Text-to-Image tasks) into a compressed latent representation. This latent representation is subsequently used by the UNet model to learn how to denoise the image as part of the reverse diffusion process.
x (pixel space) -> z (latent space)

The encoder's role is to reduce the high-dimensional input data into a lower-dimensional representation. Notably, at each stage, the number of channels in the data increases while its height and width decrease.

Initially, each pixel in the image has three channels (RGB). Through the application of convolutions, the spatial dimensions of the image are reduced while the number of features per pixel increases. Consequently, each pixel in the transformed image captures more information than in the original.

'''



class VAE_Encoder(nn.Sequential):
    '''
    This class defines the Variational Autoencoder (VAE) Encoder. It inherits from nn.Sequential, 
    stacking various convolutional and residual blocks to transform an input image into a latent 
    space representation. The encoder reduces the spatial dimensions while increasing the depth 
    (number of channels), thereby compressing the information.

    Attributes:
        The layers of the encoder include:
        - Convolutional layers with increasing depth
        - Residual blocks to maintain information
        - Attention blocks for better feature representation
        - Group normalization and activation functions for stability and non-linearity
    '''
    def __init__(self):
        '''
        Initializes the VAE_Encoder with a sequence of layers. Each layer performs specific 
        transformations, including convolutions, residual connections, attention mechanisms, 
        and normalization.

        The layers defined are:
        - Convolutional layers to downsample the image and increase channel depth
        - VAE_ResidualBlock to preserve information and aid in gradient flow
        - VAE_AttentionBlock to capture dependencies across different parts of the image
        - GroupNorm for stabilizing the training and improving convergence
        - SiLU activation function for non-linear transformations
        '''
        super().__init__(

            #(batch size, channel, height, width) -> (batch size, 128, height, width)
            nn.conv2d(3, 128, kernel_size = 3, padding = 1),

            #(batch size, 128, height, width) -> (batch size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            #(batch size, 128, height, width) -> (batch size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            #(batch size, 128, height, width) -> (batch size, 128, height/2, width/2)
            nn.conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),

            #(batch size, 128, height/2, width/2) -> (batch size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            #(batch size, 256, height/2, width/2) -> (batch size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),

            #(batch size, 256, height/2, width/2) -> (batch size, 256, height/4, width/4)
            nn.conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),

            #(batch size, 256, height/4, width/4) -> (batch size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            #(batch size, 512, height/4, width/4) -> (batch size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            #(batch size, 512, height/4, width/4) -> (batch size, 512, height/8, width/8)
            nn.conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            #(batch size, 512, height/8, width/8) -> (batch size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batch size, 512, height/8, width/8) -> (batch size, 512, height/8, width/8)
            VAE_AttentionBlock(512),

            #(batch size, 512, height/8, width/8) -> (batch size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),

            #(batch size, 512, height/8, width/8) -> (batch size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),

            #(batch size, 512, height/8, width/8) -> (batch size, 512, height/8, width/8)
            nn.SiLU(), # Sigmoid Linear Unit Activation function (swish function) 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            #(batch size, 512, height/8, width/8) -> (batch size, 8, height/8, width/8)
            nn.conv2d(512, 8, kernel_size = 3, padding = 1),

            #(batch size, 8, height/8, width/8) -> (batch size, 8, height/8, width/8)
            nn.conv2d(8, 8, kernel_size = 1, padding = 0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the VAE_Encoder.

        Inputs:
        - x (torch.Tensor): The input tensor with shape (batch_size, channel, height, width).
        - noise (torch.Tensor): Random noise tensor with shape (batch_size, 4, height/8, width/8).

        Purpose:
        The forward method processes the input image through the defined layers sequentially. 
        It performs padding during downsampling to maintain spatial dimensions, applies 
        convolutions and residual connections, and finally splits the output into mean and 
        log_variance. It then calculates the standard deviation and generates the latent 
        representation by sampling from a normal distribution scaled by the mean and variance.

        Output:
        - torch.Tensor: The latent representation of the input image with the same spatial dimensions 
          as the input noise but with compressed feature representation.
        '''
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, 4, height/8, width/8)

        # run sequentially all these models
        for module in self:
            if getattr(module, 'stride', None) == (2, 2): # padding at downsamplings should be asynnetric
                #  Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).

                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # (batch_size, 8, height/8, width/8) -> 2 tensors of shape (batch size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)
        
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        #(batch size, 4, height/8, width/8) -> (batch size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        #(batch size, 4, height/8, width/8) -> (batch size, 4, height/8, width/8)
        variance = log_variance.exp()

        # (batch size, 4, height/8, width/8) -> (batch size, 4, height/8, width/8)
        stdev = variance.sqrt()

        #z = N(0, 1)
        #X = N(mean, variance)
        #X = mean + stdev*z
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev*noise

        # scale by a constant 
        # reference: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215

        return x
