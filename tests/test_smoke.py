import unittest
from unittest import mock

import torch
import torch.nn as nn

from dataset import FrequencyMask, TimeMask
import model_slot
import train_slot


class ModelModeTest(unittest.TestCase):
    def test_eval_selects_inference_mode(self):
        model = model_slot.mymodel.__new__(model_slot.mymodel)
        nn.Module.__init__(model)
        model.mode = "train"

        self.assertIs(model.eval(), model)
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "eval")

        self.assertIs(model.train(), model)
        self.assertTrue(model.training)
        self.assertEqual(model.mode, "train")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, frame, spec):
        self.calls += 1
        loss = self.weight.square() + frame.mean() * 0 + spec.mean() * 0
        return loss, loss, loss, loss


class TrainingLoopTest(unittest.TestCase):
    def test_train_consumes_all_batches(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        sample = (
            torch.zeros(2, 3, 4, 4),
            torch.zeros(2, 1, 4, 4),
            {},
            ["a", "b"],
            ["_", "_"],
        )
        loader = [sample, sample, sample]
        args = mock.Mock(
            gpu=None,
            warmup=-1,
            epochs=1,
            lam1=0.1,
            lam2=0.1,
            lam3=100.0,
        )

        with mock.patch.object(train_slot.wandb, "log"):
            train_slot.train(loader, model, optimizer, scaler, 0, args)

        self.assertEqual(model.calls, len(loader))


class MetricNamingTest(unittest.TestCase):
    def test_validate_uses_paper_metric_names(self):
        model = TinyModel()
        object_saliency_model = mock.Mock()
        args = mock.Mock(trainset="flickr_10k")
        returned_metrics = (
            0.10, 0.11,  # AUD
            0.20, 0.21,  # IMG_QUERY
            0.30, 0.31,  # IQR
            0.40, 0.41,  # OBJ_PRIOR
            0.50, 0.51,  # OGL
            0.60, 0.61,  # EXTRA_IQR_OGL
        )

        with mock.patch.object(
            train_slot.test_model,
            "validate_img_aud",
            return_value=returned_metrics,
        ), mock.patch.object(train_slot.torch, "save"), mock.patch.object(
            train_slot.wandb, "log"
        ) as wandb_log, mock.patch("builtins.print") as print_mock:
            best = train_slot.validate(
                [], "flickr", model, object_saliency_model, 0,
                [0.0] * 10, "/tmp/jsa-test", args,
            )

        output = "\n".join(" ".join(map(str, call.args)) for call in print_mock.call_args_list)
        self.assertIn("IQR_flickr/cIoU", output)
        self.assertIn("OGL_flickr/cIoU", output)
        self.assertNotIn("AUD_ORIG_OBJ", output)
        self.assertNotIn("ALL_COMBINED", output)
        self.assertEqual(best[4], 0.30)
        self.assertEqual(best[6], 0.50)

        logged_metrics = wandb_log.call_args.args[0]
        self.assertEqual(logged_metrics["IQR_flickr/cIoU"], 0.30)
        self.assertEqual(logged_metrics["OGL_flickr/cIoU"], 0.50)
        self.assertEqual(logged_metrics["EXTRA_IQR_OGL_flickr/cIoU"], 0.60)


class AudioMaskTest(unittest.TestCase):
    def test_frequency_mask_uses_frequency_axis(self):
        value = torch.ones(1, 4, 7)
        with mock.patch("dataset.random.randrange", side_effect=[3, 1]):
            masked = FrequencyMask(max_width=2, use_mean=False)(value)
        self.assertEqual(masked[:, 3:4, :].count_nonzero().item(), 0)

    def test_time_mask_uses_time_axis(self):
        value = torch.ones(1, 4, 7)
        with mock.patch("dataset.random.randrange", side_effect=[6, 1]):
            masked = TimeMask(max_width=2, use_mean=False)(value)
        self.assertEqual(masked[:, :, 6:7].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
