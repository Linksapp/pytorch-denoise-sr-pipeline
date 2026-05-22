import torch.nn as nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  Channel Attention Mechanisms (like CBAM) uses AdaptiveAvgPool2d(1) and AdaptiveMaxPool2d(1) to generate compact descriptors that are then passed through a lightweight MLP to compute attention weights.
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.shared_mlp[:3](avg_out)
        max_out = self.shared_mlp[:3](max_out)
        out = self.shared_mlp[3](avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        concat = torch.cat([avg_out, max_out], dim=1)
        weights = self.sigmoid(self.conv(concat))
        
        return x * weights

class DenoiseResidualBlock(nn.Module):
    def __init__(self, channels, reduction=8, dilation=1, attention_type='CBAM', res_scale=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        self.attention_type = attention_type
        self.res_scale = res_scale
    

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.attention_type in ['CBAM', 'Channel']:
            out = self.channel_attention(out) 
        if self.attention_type in ['CBAM', 'Spatial']:
            out = self.spatial_attention(out)
        
        return out * self.res_scale + residual # skip-connection + residual scaling

class EnhancedDenoiser(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, reduction=1, num_blocks=1, num_filters=64, dilation=1, attention_type='CBAM', res_scale=1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_filters, 3, padding=1)
        if dilation == 1:
            self.body = nn.Sequential(*[DenoiseResidualBlock(num_filters, reduction, attention_type=attention_type,  res_scale=res_scale) for _ in range(num_blocks)])
        else:
            body_layers = []
            for i in range(num_blocks):
                d = 1 if i % 2 == 0 else dilation
                body_layers.append(DenoiseResidualBlock(num_filters, reduction, d, attention_type, res_scale))
            self.body = nn.Sequential(*body_layers)

        self.tail = nn.Conv2d(num_filters, out_channels, 3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        x_tail = self.tail(x_body)
        return x[:, :3, :, :] - x_tail
    
# num_blocks - 16 (больше блоков - глубже сеть) 
# reduction - 8 (больше внимания)
# Multi-scale features