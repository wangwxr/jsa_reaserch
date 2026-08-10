# JSA reproduction on Flickr-SoundNet

The first reproduction target is Flickr-SoundNet-10k training followed by
evaluation on the 250 annotated Flickr-SoundNet-Test pairs. VGGSound is not
required for this route.

## Data

The local preparation command fully extracts the official SoundNet archives
and then creates flat 10k and 144k views using symlinks:

```bash
bash scripts/prepare_flickr_all.sh \
  /data/wxr/datasets/FlickrSoundNet \
  /data/wxr/audio_video/EZ-VSL/metadata
```

The EZ-VSL Flickr manifests span both SoundNet's `train` and `val` indexes.
The view builder therefore indexes both sources; the verified local views
contain exactly 10,000 and 144,000 paired image/audio IDs.

The resulting layout is:

```text
FlickrSoundNet/
├── extracted/                 # complete lists, frames and mp3 archives
├── prepared/
│   ├── jsa_flickr_10k/        # flat symlink view
│   └── jsa_flickr_144k/       # flat symlink view
└── test/Dataset/              # original 5k test package; JSA uses 250 IDs
```

For the four SoundNet frame candidates, the view prefers frame 8, then 13,
3 and 18. This selects the candidate closest to the middle of a nominal
20-second clip. The loader extracts the middle five seconds of audio and
constructs a `1 x 257 x 501` log spectrogram at 16 kHz.

## Smoke tests

```bash
python -m unittest discover -s tests -v
```

## Train and evaluate

```bash
bash scripts/train_flickr.sh 10k 0 jsa_flickr_10k
bash scripts/test_flickr.sh jsa_flickr_10k 0
```

The 144k command is:

```bash
bash scripts/train_flickr.sh 144k 0 jsa_flickr_144k
```

## Baseline

The local baseline keeps JSA's two ResNet-18 encoders and replaces joint slot
attention with global-audio/local-image MIL-InfoNCE. It therefore isolates the
effect of slots, reconstruction and cross-modal attention matching while using
the same data, optimizer and evaluator. It has 22.87M parameters versus
31.82M for the training-time JSA model, matching the model sizes in the paper:

```bash
bash scripts/train_baseline_flickr.sh 10k 0 baseline_av_mil_flickr_10k
bash scripts/test_flickr.sh baseline_av_mil_flickr_10k 0
```

The paper reports the following Flickr-SoundNet-Test targets:

| Train split | IQR cIoU | IQR AUC | With OGL cIoU | With OGL AUC |
|---|---:|---:|---:|---:|
| Flickr-10k | 85.20 | 65.26 | 87.60 | 66.50 |
| Flickr-144k | 86.00 | 65.16 | 89.20 | 64.50 |

Runtime metric labels follow the paper terminology:

- `AUD`: raw cross-modal audio-query localization.
- `IMG_QUERY`: image target-query prior used by IQR.
- `IQR`: Image-Query based Refinement reported in the paper.
- `OBJ_PRIOR`: external object-saliency prior used by OGL.
- `OGL`: Object-Guided Localization reported in the paper.
- `EXTRA_IQR_OGL`: an additional three-way fusion from the released code;
  this metric is not part of Table 1.

## Upstream blockers fixed locally

- Evaluation now switches `mymodel` to its inference branch.
- Training no longer stops after the first batch of every epoch.
- Only the selected evaluation dataset is constructed during training.
- Dataset roots are read from CLI arguments instead of the author's machine.
- Training accepts mp3/wav directly as well as precomputed NumPy spectra.
- Frequency and time masking now sample offsets from their intended axes.

One paper/code discrepancy remains explicit: the paper states IQR alpha=0.6,
while the released shell scripts use `--alpha 0.4` for all refinements. The
local run scripts initially preserve the released-code value; both values
should be reported as a small evaluation ablation before claiming exact
reproduction.
