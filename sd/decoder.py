import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

'''
    ### decoder.py Explained ###

This script contains the code for the VAE_AttentionBlocks, VAE_ResidualBlock and the actual decoder portion aswell. 

We are essentially do the opposite of the encoder.py We are reducing the number of the channels and at the same time we will increase the size of the image from latent representation back to pixel space.  


'''


class VAE_AttentionBlock(nn.Module):
    '''
    
    '''
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        #x: (batch size, features, height, width)

        residue = x

        n, c, h, w = x.shape

        #(batch size, features, height, width) -> (batch size, features, height*width)
        x = x.view((n, c, h*w))

        #(batch size, features, height*width) -> (batch size, height*width, features)
        # each pixel becomes a feature of size features, the sequence length is height*width, 
        x = x.transpose(-1, -2)

        # perform self attention without mask
        # (batch size, height*width, features) -> (batch size, height*width, features)
        x = self.attention(x)

        # (batch size, height*width, features) -> (batch size, features, height*width)
        x = x.transpose(-1, -2)

        #(batch size, features, height*width) -> (batch size, features, height, width)
        x = x.view((n, c, h, w))

        x += residue
        

        return x

class VAE_ResidualBlock(nn.Module):
    '''

    This is made up of normalizations and convolutions.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)

    
    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        residue = x
        #(batch_size, in_channels, height, width) -> (batch__size, in_channels, height, width)
        x = self.groupnorm_1(x)

        #(batch_size, in_channels, height, width) -> (batch__size, in_channels, height, width)
        x = F.silu(x)

        #(batch__size, in_channels, height, width) -> (batch__size, in_channels, height, width)
        x = self.conv_1(x)

        #(batch__size, in_channels, height, width) -> (batch__size, in_channels, height, width)
        x = self.groupnorm_2(x)
        
        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    

class VAE_Decoder(nn.Sequential):
    '''
    
    '''
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),


            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256),
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width )
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),

            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width )
            VAE_ResidualBlock(256, 128),
             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width )
            VAE_ResidualBlock(128, 128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width )
            VAE_ResidualBlock(128, 128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width )
            nn.GroupNorm(32, 128),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width )
            nn.SiLU(),
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width )
            nn.Conv2d(128, 3, kernel_Size = 3, padding = 1),

        )

    def forward(self, x ):
        # x: (batch_size, 4, height/8, width/8)
    
        # remvoe the scaling added by encoder

        x /= 0.18215

        for module in self:
            x = module(x)e

        # (batch_size, 3, height, width)
        return x