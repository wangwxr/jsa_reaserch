"""ResNet-18 exposing only projected L3/L4 features at 7x7."""

import torch

from multi_level_resnet import IMAGENET_RESNET18_URL, MultiLevelResNet18


class L3L4ResNet18(MultiLevelResNet18):
    """The v1.1 backbone without its L2 projection/output branch."""

    def __init__(self, output_dim=512):
        super().__init__(output_dim=output_dim)
        del self.proj2

    def forward(self, image):
        _, layer3, layer4 = self.forward_feature_maps(image)
        level3 = self._pool_to_7x7(self.proj3(layer3))
        level4 = self._pool_to_7x7(self.proj4(layer4))
        return self._to_tokens(level3), self._to_tokens(level4)


def resnet18_l3_l4(pretrained=True, output_dim=512):
    model = L3L4ResNet18(output_dim=output_dim)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            IMAGENET_RESNET18_URL, progress=False
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("proj3.", "proj4."))
        ]
        unexpected = set(incompatible.unexpected_keys)
        if invalid_missing or unexpected != {"fc.weight", "fc.bias"}:
            raise RuntimeError(
                "Unexpected ImageNet ResNet-18 loading result: "
                f"missing={invalid_missing}, "
                f"unexpected={sorted(unexpected)}"
            )
    return model
