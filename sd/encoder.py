import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

'''
    ### encoder.py Explained ### 

This script contains the encoder part of the Variational Auto Encoder (VAE) and also the sampling part so when you take the image run it through the encoder to make it smaller you obtain a mean and a variance. Then we sample from the distribution to give us the mean and the variance.

The main purpose of the encoder.py is to encode an image(ImageToImage)/random noise(TextToImage) into a compressed representation of this image/randomNoise.
This is so we can take this new latent representation and run it through the UNet model to learn how to denoise the image as part of the reverse process.
x (pixel space)-> z(latent space)

The encoders job is to reduce the higher dimensional representation of the data into a lower dimensional representation of the data.

Interestingly at each step, we are increasing the number of channels in the data but reducing its height and width

so we start for each pixel with 3 channels (red,blue,green RGB) but then by using convolutions we reduce the size of the image but simaltenously we increase the number of features that each pixels represents.
Hence each pixel will actually capture more data than originally. 

'''



class VAE_Encoder(nn.Sequential):
    def __init__(self):
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

