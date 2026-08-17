"""Conventional no-slot audio-visual MIL baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet
import utils


class NoSlotAVBaseline(nn.Module):
    """Global-audio/local-image contrastive SSL without Slot Attention."""

    def __init__(self, args):
        super().__init__()
        self.tau = args.tau
        self.k = args.reciprocal_k
        self.imgnet = resnet.resnet(
            pretrained=True,
            modal="vision",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )
        self.audnet = resnet.resnet(
            pretrained=False,
            modal="audio",
            dropout_rate=0.0,
            output_dim=args.out_dim,
            fourth_stride=2,
        )

    def encode(self, image, audio):
        image_features = F.normalize(self.imgnet(image), dim=1)
        audio_features = self.audnet(audio)
        audio_features = F.adaptive_max_pool1d(
            audio_features, 1
        ).squeeze(-1)
        audio_features = F.normalize(audio_features, dim=1)
        return image_features, audio_features

    @staticmethod
    def _paired_image_representation(image_features, audio_features):
        paired_similarity = torch.einsum(
            "bchw,bc->bhw", image_features, audio_features
        )
        max_indices = paired_similarity.flatten(start_dim=1).argmax(dim=1)
        image_tokens = image_features.flatten(start_dim=2).transpose(1, 2)
        batch_indices = torch.arange(
            image_features.shape[0], device=image_features.device
        )
        return image_tokens[batch_indices, max_indices]

    def forward_train(self, image, audio):
        image_features, audio_features = self.encode(image, audio)

        # Conventional MIL: select the most audio-relevant local image patch.
        spatial_logits = torch.einsum(
            "nchw,mc->nmhw", image_features, audio_features
        )
        logits = spatial_logits.flatten(start_dim=2).max(dim=2).values

        image_representation = self._paired_image_representation(
            image_features, audio_features
        )
        _, _, reciprocal = utils.get_potential_false_negative(
            image_representation.detach().unsqueeze(1),
            audio_features.detach().unsqueeze(1),
            k=self.k,
        )
        logits = logits.masked_fill(reciprocal == False, -float("inf"))

        labels = torch.arange(image.shape[0], device=image.device)
        info_loss = F.cross_entropy(logits / self.tau, labels)
        info_loss = info_loss + F.cross_entropy(
            logits.transpose(0, 1) / self.tau, labels
        )

        # Interface compatibility only: these losses do not exist in B0.
        not_applicable = info_loss.new_zeros(())
        return info_loss, not_applicable, not_applicable, not_applicable

    def forward_eval(self, image, audio):
        image_features, audio_features = self.encode(image, audio)
        return torch.einsum(
            "bchw,bc->bhw", image_features, audio_features
        ).unsqueeze(1)

    def forward(self, image, audio):
        if self.training:
            return self.forward_train(image, audio)
        return self.forward_eval(image, audio)
