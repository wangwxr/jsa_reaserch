#!/usr/bin/env python3
"""Precompute Flickr spectra aligned to the selected 8 s or 3 s frame."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio


FRAME_TIMES = {
    "00000008.jpg": 8.0,
    "00000003.jpg": 3.0,
}
FRAME_PREFERENCE = (
    "00000008.jpg",
    "00000013.jpg",
    "00000003.jpg",
    "00000018.jpg",
)
TARGET_SAMPLE_RATE = 16000
_SPECTROGRAM = None
_RESAMPLERS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use a 5 s window centered at 8 s for 00000008.jpg samples. "
            "For 00000003.jpg fallbacks, use the first 5 s and repeat short "
            "audio until the window is full."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-dir", required=True)
    frame_source = parser.add_mutually_exclusive_group(required=True)
    frame_source.add_argument("--frame-view-dir")
    frame_source.add_argument(
        "--lists-dir",
        help="SoundNet lists directory; avoids a slow per-symlink scan.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Process only this ID; may be supplied multiple times for checks.",
    )
    return parser.parse_args()


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def load_manifest_ids(path):
    ids = []
    seen = set()
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            sample_id = row[0].strip()
            if sample_id and sample_id not in seen:
                ids.append(sample_id)
                seen.add(sample_id)
    return ids


def index_audio_files(audio_dir):
    by_id = {}
    priority = {".wav": 0, ".mp3": 1}
    with os.scandir(audio_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            sample_id, extension = os.path.splitext(entry.name)
            extension = extension.lower()
            if extension not in priority:
                continue
            previous = by_id.get(sample_id)
            if previous is None or priority[extension] < priority[previous[0]]:
                by_id[sample_id] = (extension, entry.path)
    return {sample_id: value[1] for sample_id, value in by_id.items()}


def selected_frame_name(frame_view_dir, sample_id):
    path = Path(frame_view_dir) / f"{sample_id}.jpg"
    if not path.is_symlink():
        raise RuntimeError(
            f"Selected-frame view must contain symlinks, but this is not one: {path}"
        )
    frame_name = Path(os.readlink(path)).name
    if frame_name not in FRAME_TIMES:
        raise RuntimeError(
            f"Unsupported selected frame for {sample_id}: {frame_name}"
        )
    return frame_name


def index_selected_frame_names(lists_dir, sample_ids):
    wanted = {sample_id.encode(): sample_id for sample_id in sample_ids}
    ranks = {
        frame_name.encode(): rank
        for rank, frame_name in enumerate(FRAME_PREFERENCE)
    }
    selected = {}
    list_paths = sorted(Path(lists_dir).glob("*_frames4_*.txt"))
    if not list_paths:
        raise RuntimeError(f"No *_frames4_*.txt files found in {lists_dir}")

    for list_path in list_paths:
        with list_path.open("rb", buffering=4 * 1024 * 1024) as handle:
            for line in handle:
                source_path = line.split(None, 1)[0]
                _, marker, relative_path = source_path.partition(b"/frames/")
                if not marker:
                    continue
                frame_root, frame_name = relative_path.rsplit(b"/", 1)
                rank = ranks.get(frame_name)
                if rank is None:
                    continue
                video_id = frame_root.rsplit(b"/", 1)[1]
                if video_id.endswith(b".mp4"):
                    video_id = video_id[:-4]
                sample_id = wanted.get(video_id)
                if sample_id is None:
                    continue
                current = selected.get(sample_id)
                if current is None or rank < current[0]:
                    selected[sample_id] = (rank, frame_name.decode())

    missing = [sample_id for sample_id in sample_ids if sample_id not in selected]
    if missing:
        raise RuntimeError(
            f"Missing selected frame for {len(missing)}/{len(sample_ids)} IDs; "
            f"examples: {', '.join(missing[:5])}"
        )
    return {sample_id: value[1] for sample_id, value in selected.items()}


def init_worker():
    global _SPECTROGRAM, _RESAMPLERS
    torch.set_num_threads(1)
    _SPECTROGRAM = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=512,
        hop_length=160,
        center=True,
        power=2.0,
    )
    _RESAMPLERS = {}


def resample(waveform, source_rate):
    if source_rate == TARGET_SAMPLE_RATE:
        return waveform
    resampler = _RESAMPLERS.get(source_rate)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_rate, new_freq=TARGET_SAMPLE_RATE
        )
        _RESAMPLERS[source_rate] = resampler
    return resampler(waveform)


def extract_window(waveform, frame_name, duration):
    required = int(round(TARGET_SAMPLE_RATE * duration))
    available = waveform.shape[1]
    if available == 0:
        raise RuntimeError("decoded audio is empty")

    if frame_name == "00000008.jpg":
        if available < required:
            repeats = (required + available - 1) // available
            return (
                waveform.repeat(1, repeats)[:, :required],
                True,
                0.0,
                False,
            )
        desired_start = int(
            round((FRAME_TIMES[frame_name] - duration / 2.0) * TARGET_SAMPLE_RATE)
        )
        # Keep a continuous five-second crop when the audio ends before 10.5 s.
        # This is the closest valid boundary crop and still contains the 8 s frame.
        start = min(desired_start, available - required)
        boundary_shifted = start != desired_start
        return (
            waveform[:, start : start + required],
            False,
            start / TARGET_SAMPLE_RATE,
            boundary_shifted,
        )

    window = waveform[:, :required]
    repeated = window.shape[1] < required
    if repeated:
        repeats = (required + window.shape[1] - 1) // window.shape[1]
        window = window.repeat(1, repeats)
    return window[:, :required], repeated, 0.0, False


def convert_one(task):
    sample_id, input_path, output_path, frame_name, duration = task
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return sample_id, "skipped", frame_name, False, False, ""

    temporary_path = None
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = resample(waveform, sample_rate)
        window, repeated, _, boundary_shifted = extract_window(
            waveform, frame_name, duration
        )
        spectrogram = _SPECTROGRAM(window)
        array = spectrogram.detach().cpu().numpy().astype(np.float32, copy=False)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(output_path),
            prefix=f".{sample_id}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary_path, output_path)
        return (
            sample_id,
            "converted",
            frame_name,
            repeated,
            boundary_shifted,
            "",
        )
    except Exception as exc:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        return (
            sample_id,
            "failed",
            frame_name,
            False,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.duration <= 0 or args.duration > 2 * FRAME_TIMES["00000008.jpg"]:
        raise ValueError("--duration must be greater than 0 and at most 16 seconds")

    manifest_ids = load_manifest_ids(args.manifest)
    if args.sample_ids:
        requested = set(args.sample_ids)
        unknown = requested.difference(manifest_ids)
        if unknown:
            raise RuntimeError(f"IDs absent from manifest: {', '.join(sorted(unknown))}")
        manifest_ids = [sample_id for sample_id in manifest_ids if sample_id in requested]
    elif args.limit is not None:
        manifest_ids = manifest_ids[: args.limit]

    audio_files = index_audio_files(args.audio_dir)
    missing_audio = [sample_id for sample_id in manifest_ids if sample_id not in audio_files]
    if missing_audio:
        raise RuntimeError(
            f"Missing source audio for {len(missing_audio)}/{len(manifest_ids)} IDs; "
            f"examples: {', '.join(missing_audio[:5])}"
        )

    if args.lists_dir:
        frame_names = index_selected_frame_names(args.lists_dir, manifest_ids)
    else:
        frame_names = {
            sample_id: selected_frame_name(args.frame_view_dir, sample_id)
            for sample_id in manifest_ids
        }
    unsupported_frames = sorted(set(frame_names.values()).difference(FRAME_TIMES))
    if unsupported_frames:
        raise RuntimeError(
            "The selected image view contains unsupported frame times: "
            + ", ".join(unsupported_frames)
        )
    frame_counts = {
        name: sum(frame_name == name for frame_name in frame_names.values())
        for name in FRAME_TIMES
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            sample_id,
            audio_files[sample_id],
            str(output_dir / f"{sample_id}.npy"),
            frame_names[sample_id],
            args.duration,
        )
        for sample_id in manifest_ids
    ]
    tasks.sort(key=lambda task: task[1])

    print(
        f"Spectrograms: {len(tasks)}, workers: {args.workers}, "
        f"frame8={frame_counts['00000008.jpg']}, "
        f"frame3={frame_counts['00000003.jpg']}, output: {output_dir}",
        flush=True,
    )
    converted = 0
    skipped = 0
    repeated_frame3 = 0
    repeated_frame8 = 0
    shifted_frame8 = 0
    failures = []
    start = time.monotonic()
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers, initializer=init_worker) as pool:
        results = pool.imap_unordered(convert_one, tasks, chunksize=1)
        for completed, result in enumerate(results, 1):
            sample_id, status, frame_name, repeated, boundary_shifted, message = result
            if status == "converted":
                converted += 1
                if frame_name == "00000003.jpg" and repeated:
                    repeated_frame3 += 1
                if frame_name == "00000008.jpg" and repeated:
                    repeated_frame8 += 1
                if frame_name == "00000008.jpg" and boundary_shifted:
                    shifted_frame8 += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append((sample_id, frame_name, message))
                if len(failures) <= 5:
                    print(
                        f"Failure {len(failures)}: {sample_id} "
                        f"({frame_name}): {message}",
                        flush=True,
                    )

            if completed % args.report_every == 0 or completed == len(tasks):
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(tasks) - completed) / rate if rate else 0.0
                print(
                    f"Progress {completed}/{len(tasks)} "
                    f"({100.0 * completed / len(tasks):.2f}%); "
                    f"converted={converted}, skipped={skipped}, "
                    f"repeat3={repeated_frame3}, repeat8={repeated_frame8}, "
                    f"shift8={shifted_frame8}, "
                    f"failed={len(failures)}; "
                    f"rate={rate:.2f} files/s; ETA={format_duration(eta)}",
                    flush=True,
                )

    elapsed = time.monotonic() - start
    print(
        f"Finished in {format_duration(elapsed)}: converted={converted}, "
        f"skipped={skipped}, repeat3={repeated_frame3}, "
        f"repeat8={repeated_frame8}, "
        f"shift8={shifted_frame8}, failed={len(failures)}",
        flush=True,
    )
    if failures:
        failure_path = output_dir / "frame_aligned_failures.csv"
        with failure_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "frame_name", "error"])
            writer.writerows(failures)
        raise RuntimeError(
            f"Frame-aligned precomputation failed for {len(failures)} files; "
            f"see {failure_path}"
        )

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "requested": len(tasks),
        "converted": converted,
        "skipped": skipped,
        "frame8": frame_counts["00000008.jpg"],
        "frame3": frame_counts["00000003.jpg"],
        "repeated_frame3_this_run": repeated_frame3,
        "repeated_short_frame8_this_run": repeated_frame8,
        "boundary_shifted_frame8_this_run": shifted_frame8,
        "duration_seconds": args.duration,
        "frame8_audio_window_seconds": [
            FRAME_TIMES["00000008.jpg"] - args.duration / 2.0,
            FRAME_TIMES["00000008.jpg"] + args.duration / 2.0,
        ],
        "frame8_boundary_rule": (
            "use [5.5, 10.5] when available; otherwise shift left to the "
            "last continuous five seconds; repeat from the beginning only "
            "when the entire audio is shorter than five seconds"
        ),
        "frame3_audio_rule": "first five seconds; repeat only when shorter",
        "spectrogram": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "n_fft": 512,
            "win_length": 512,
            "hop_length": 160,
            "power": 2.0,
            "log_applied": False,
        },
    }
    report_path = output_dir.parent / "frame_aligned_preparation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
