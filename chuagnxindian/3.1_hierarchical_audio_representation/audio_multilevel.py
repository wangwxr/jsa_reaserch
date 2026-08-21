"""Audio ResNet18 with an exact baseline A4 path plus a native A3 readout."""

from __future__ import annotations

import torch.nn as nn

import resnet


class AudioResNet18MultiLevel(resnet.ResNet):
    """Return raw and projected layer3/layer4 audio representations."""

    def __init__(self, output_dim: int = 512, fourth_stride: int = 2):
        if output_dim != 512:
            raise ValueError("Experiment 3.1 requires output_dim=512")
        super().__init__(
            block=resnet.BasicBlock,
            layers=[2, 2, 2, 2],
            modal="audio",
            dropout_rate=0.0,
            output_dim=output_dim,
            fourth_stride=fourth_stride,
        )
        self.aud_proj3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.AdaptiveMaxPool2d((1, None)),
            nn.Flatten(start_dim=2),
        )
        nn.init.kaiming_normal_(
            self.aud_proj3[0].weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        raw_a3 = self.layer3(x)
        raw_a4 = self.layer4(raw_a3)
        a3_feature = self.aud_proj3(raw_a3)
        a4_feature = self.proj(raw_a4)
        return {
            "raw_a3": raw_a3,
            "raw_a4": raw_a4,
            "a3_feature": a3_feature,
            "a4_feature": a4_feature,
        }


def audio_resnet18_multilevel(output_dim: int = 512, fourth_stride: int = 2):
    return AudioResNet18MultiLevel(
        output_dim=output_dim,
        fourth_stride=fourth_stride,
    )
