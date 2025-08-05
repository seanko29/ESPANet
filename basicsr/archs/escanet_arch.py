import math
import numbers
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from einops import rearrange
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from basicsr.utils.registry import ARCH_REGISTRY




class ESAB(nn.Module):
    def __init__(self, alpha):
        super(ESAB, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        k = x.mean(dim=[-1, -2], keepdim=True)  # N,C,1,1
        kd = torch.sqrt((k - k.mean(dim=1, keepdim=True)).pow(2).sum(dim=1, keepdim=True))  # N,1,1,1
        Qd = torch.sqrt((x - x.mean(dim=1, keepdim=True)).pow(2).sum(dim=1, keepdim=True))  # N,1,H,W

        C_Qk = ((x - x.mean(dim=1, keepdim=True)) * (k - k.mean(dim=1, keepdim=True))).sum(dim=1, keepdim=True) / (Qd * kd)  # N,1,H,W

        A = (1 - torch.sigmoid(C_Qk)) ** self.alpha  # N,1,H,W

        out = x * A  # N,C,H,W
        return out
    

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class ExtraDW(nn.Module):
    def __init__(self, channels, kernel_size, stride=1):
        super(ExtraDW, self).__init__()
        self.dw = conv_2d(channels, channels, kernel_size=kernel_size, stride=stride, groups=channels)
        
    def forward(self, x):
        return self.dw(x)
    
    
            
class SEM(nn.Module):
    def __init__(self, channel, splits=2):
        super().__init__()
        self.splits = splits
        chuncks_channel = channel

        # dw 4,1,3,3
        self.depthwise = nn.Conv2d(chuncks_channel, chuncks_channel, 3, 1, 1, groups=chuncks_channel)
        self.pointwise = nn.Conv2d(chuncks_channel, chuncks_channel, 1, 1, 0)
        self.activation = nn.GELU()
                
        self.last_pointwise = nn.Conv2d(channel, channel, 1, 1, 0)
        self.ccm = nn.Conv2d(channel, channel*2, 1, 1, 0) # cross-channel mixing
        self.attention = ESAB(0.5)
        self.layernorm = LayerNorm(channel)
        
        self.eca_layer = eca_layer(channel)
        
    def forward(self, x):
        h, w = x.size()[-2:]
        new_h, new_w = h // 8, w // 8
        
        x = self.ccm(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = F.adaptive_max_pool2d(x1, (new_h, new_w))
        x1 = self.depthwise(x1)
        x1 = self.pointwise(x1)
        x1 = F.interpolate(x1, size=(h,w), mode='nearest')
        
        x2 = self.eca_layer(x2)
                
        # concatenated = torch.cat([x1, x2], dim=1)
        x_f = x1 + x2
        concat_output = self.layernorm(self.attention(x_f))
        output = self.activation(self.last_pointwise(concat_output)) 
        
           
        # combined = self.activation(concat_output) * x
        # output = self.activation(self.last_pointwise(concat_output)) * x
        
        return output
    
class CEM(nn.Module):
    def __init__(self, channel, expand_rate=2.0, down_rate=0.25):
        super().__init__()
        hidden_dim = int(channel * expand_rate)
        down_dim = int(hidden_dim * down_rate)
        
        self.hidden_dim = hidden_dim
        self.down_dim = down_dim
        channel_chunk = channel
        
        self.proj = nn.Conv2d(channel, channel*2, 1, 1, 0)
        self.act = nn.GELU()
        
        self.path_q =  nn.Sequential(
            nn.Conv2d(channel_chunk, channel_chunk, 3, 1, 1, groups=channel_chunk),
            nn.Conv2d(channel_chunk, hidden_dim, 1, 1, 0),
            nn.Conv2d(hidden_dim, channel_chunk, 1, 1, 0),
            nn.GELU()
        )
        
        self.proj_out = nn.Conv2d(channel_chunk, channel, 1, 1, 0)

    def forward(self, x):
        identity = x
        x = self.proj(x)  # cross-channel information mixing (pointwise conv1x1)
        
        # if self.training:
            # Training Mode: split and process separately
        q, p = x.chunk(2, dim=1)
        q = self.path_q(q)
        # output = torch.cat([q, p], dim=1)
        output = p * q
        # else:
        #     # Inference Mode: in-place operation on part of x
        #     q = x[:, :self.down_dim, :, :]
        #     x[:, :self.down_dim, :, :] = self.path_q(q)
        #     output = x
        
        # print(output.shape)
        output = self.proj_out(output)
        return output + identity
    
    
# Feature Enhancement Block
class UniMix(nn.Module):
    def __init__(self, channel, expand_rate=2.0, down_rate=0.25, layer_norm=False):
        super().__init__()
        
        if layer_norm:
            self.se_block = nn.Sequential(
                LayerNorm(channel),
                SEM(channel),
            )
            
            self.ce_block = nn.Sequential(
                LayerNorm(channel),
                CEM(channel, expand_rate, down_rate),
            )
        else:
            self.se_block = SEM(channel)
            self.ce_block = CEM(channel, expand_rate, down_rate)
            
    def forward(self, x):
        x = self.se_block(x) + x
        x = self.ce_block(x) + x
        return x
        
        


@ARCH_REGISTRY.register()
class ESCANet(nn.Module):
    def __init__(self, dim=64, n_blocks=12, upscaling_factor=4, expand_rate=2.0, down_rate=0.25, layer_norm=False):
        super().__init__()
        self.head = nn.Conv2d(3, dim, 3, 1, 1)
        
        self.unimix = nn.Sequential(*[UniMix(dim, expand_rate, down_rate, layer_norm) for _ in range(n_blocks)])
        
        self.tail = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )
    def forward(self, x):
        x = self.head(x)
        x = self.unimix(x) + x
        x = self.tail(x)
        
        return x    
