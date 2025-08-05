import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class DiscriminatorVGG(nn.Module):
    """A small SRGAN‐style VGG-like discriminator."""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        layers = []
        c = in_channels
        # 1) conv block ×4 with downsampling
        for i in range(4):
            out_c = base_channels * (2**i)
            layers += [
                nn.Conv2d(c, out_c, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c = out_c
        # 2) final classifier
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
