"""ResNet-18 wrapper preserving projected layer3 at its native 14x14 grid."""

import torch
from multi_level_resnet import (
    IMAGENET_RESNET18_URL,
    MultiLevelResNet18,
)


class NativeL3MultiLevelResNet18(MultiLevelResNet18):
    """The v1.1 backbone with only the L3 pooling operation removed."""

    def forward(self, x):
        layer2, layer3, layer4 = self.forward_feature_maps(x)

        level2 = self._pool_to_7x7(self.proj2(layer2))
        level3 = self.proj3(layer3)
        level4 = self._pool_to_7x7(self.proj4(layer4))

        if level2.shape[-2:] != (7, 7):
            raise RuntimeError(f"Expected projected L2 7x7, got {level2.shape}")
        if level3.shape[-2:] != (14, 14):
            raise RuntimeError(f"Expected native projected L3 14x14, got {level3.shape}")
        if level4.shape[-2:] != (7, 7):
            raise RuntimeError(f"Expected projected L4 7x7, got {level4.shape}")

        return tuple(self._to_tokens(level) for level in (level2, level3, level4))


def resnet18_native_l3(pretrained=True, output_dim=512):
    model = NativeL3MultiLevelResNet18(output_dim=output_dim)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            IMAGENET_RESNET18_URL, progress=False
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("proj2.", "proj3.", "proj4.")
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        unexpected = set(incompatible.unexpected_keys)
        if invalid_missing or unexpected != {"fc.weight", "fc.bias"}:
            raise RuntimeError(
                "Unexpected ImageNet ResNet-18 loading result: "
                f"missing={invalid_missing}, unexpected={sorted(unexpected)}"
            )
    return model
