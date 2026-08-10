"""Dual-encoder MIL baseline with the same encoders used by JSA."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet


class AudioVisualMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tau = args.tau
        self.imgnet = resnet.resnet(True, 'vision', 0.0, args.out_dim, 2)
        self.audnet = resnet.resnet(False, 'audio', 0.0, args.out_dim, 2)

    def encode(self, image, audio):
        image_features = F.normalize(self.imgnet(image), dim=1)
        audio_features = self.audnet(audio)
        audio_features = F.adaptive_max_pool1d(audio_features, 1).squeeze(-1)
        audio_features = F.normalize(audio_features, dim=1)
        return image_features, audio_features

    def forward_train(self, image, audio):
        image_features, audio_features = self.encode(image, audio)
        spatial_logits = torch.einsum(
            'nchw,mc->nmhw', image_features, audio_features) / self.tau
        logits = spatial_logits.flatten(-2).max(dim=-1).values
        labels = torch.arange(image.shape[0], device=image.device)
        loss = F.cross_entropy(logits, labels)
        loss = loss + F.cross_entropy(logits.transpose(0, 1), labels)
        zero = loss.new_zeros(())
        return loss, zero, zero, zero

    def forward_eval(self, image, audio):
        image_features, audio_features = self.encode(image, audio)
        localization = torch.einsum(
            'bchw,bc->bhw', image_features, audio_features).unsqueeze(1)
        return localization, localization

    def forward(self, image, audio):
        if self.training:
            return self.forward_train(image, audio)
        return self.forward_eval(image, audio)
