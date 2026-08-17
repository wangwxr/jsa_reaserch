"""ResNet-18 visual encoder exposing pooled layer2/layer3/layer4 tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class MultiLevelResNet18(nn.Module):
    """JSA-compatible ResNet-18 returning three 7x7 token sequences."""

    def __init__(self, output_dim=512):
        super().__init__()
        if output_dim != 512:
            raise ValueError("MUFASA-JSA v1 fixes the visual slot dimension to 512")

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.proj2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(256, output_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(512, output_dim, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                self._norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                )
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _pool_to_7x7(feature):
        if feature.shape[-2:] == (7, 7):
            return feature
        return F.adaptive_avg_pool2d(feature, (7, 7))

    @staticmethod
    def _to_tokens(feature):
        return feature.flatten(start_dim=2).transpose(1, 2)

    def forward_feature_maps(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer2, layer3, layer4

    def forward(self, x):
        layer2, layer3, layer4 = self.forward_feature_maps(x)

        level2 = self._pool_to_7x7(self.proj2(layer2))
        level3 = self._pool_to_7x7(self.proj3(layer3))
        level4 = self._pool_to_7x7(self.proj4(layer4))

        return tuple(self._to_tokens(level) for level in (level2, level3, level4))


def resnet18_multilevel(pretrained=True, output_dim=512):
    model = MultiLevelResNet18(output_dim=output_dim)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            IMAGENET_RESNET18_URL, progress=False
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("proj2.", "proj3.", "proj4.")
        unexpected = set(incompatible.unexpected_keys)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or unexpected != {"fc.weight", "fc.bias"}:
            raise RuntimeError(
                "Unexpected ImageNet ResNet-18 loading result: "
                f"missing={invalid_missing}, unexpected={sorted(unexpected)}"
            )
    return model
