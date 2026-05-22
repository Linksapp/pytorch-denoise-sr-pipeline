import torch.nn as nn

class UpscaleResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class Upscaler(nn.Module):
    def __init__(self, scale_factor=2, in_channels=3, num_blocks=8):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.body = nn.Sequential(*[UpscaleResidualBlock(64) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(64, 64, 3, padding=1)

        self.upscale = nn.Conv2d(64, in_channels * (scale_factor ** 2), 3, padding=1) # учесть наличия nm?
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = self.conv_in(x)
        residual = x
        x = self.body(x)
        x = self.conv_out(x) + residual
        x = self.upscale(x)
        x = self.pixel_shuffle(x)
        return x