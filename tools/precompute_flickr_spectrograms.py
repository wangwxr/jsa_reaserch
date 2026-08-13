#!/usr/bin/env python3
import argparse
import csv
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_spectrogram_torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute fixed-window power spectrograms as float32 NPY files."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--crop", choices=("center", "first"), default="center")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


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
    priority = {".npy": 0, ".wav": 1, ".mp3": 2}
    with os.scandir(audio_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=True):
                continue
            sample_id, extension = os.path.splitext(entry.name)
            extension = extension.lower()
            if extension not in priority or extension == ".npy":
                continue
            previous = by_id.get(sample_id)
            if previous is None or priority[extension] < priority[previous[0]]:
                by_id[sample_id] = (extension, entry.path)
    return {sample_id: value[1] for sample_id, value in by_id.items()}


def init_worker():
    torch.set_num_threads(1)


def convert_one(task):
    sample_id, input_path, output_path, duration, crop_mode = task
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return sample_id, "skipped", ""

    temporary_path = None
    try:
        spectrogram, _ = load_spectrogram_torchaudio(
            input_path,
            dur=duration,
            rand=False,
            log_spectrogram=False,
            crop_mode=crop_mode,
        )
        array = spectrogram.detach().cpu().numpy().astype(np.float32, copy=False)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=os.path.dirname(output_path),
            prefix=f".{sample_id}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary_path, output_path)
        return sample_id, "converted", ""
    except Exception as exc:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        return sample_id, "failed", f"{type(exc).__name__}: {exc}"


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    manifest_ids = load_manifest_ids(args.manifest)
    if args.limit is not None:
        manifest_ids = manifest_ids[:args.limit]

    audio_files = index_audio_files(args.audio_dir)
    missing = [sample_id for sample_id in manifest_ids if sample_id not in audio_files]
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing source audio for {len(missing)}/{len(manifest_ids)} IDs; "
            f"examples: {examples}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            sample_id,
            audio_files[sample_id],
            str(output_dir / f"{sample_id}.npy"),
            args.duration,
            args.crop,
        )
        for sample_id in manifest_ids
    ]

    print(
        f"Spectrograms: {len(tasks)}, workers: {args.workers}, "
        f"duration: {args.duration:.1f}s, crop: {args.crop}, output: {output_dir}",
        flush=True,
    )
    converted = 0
    skipped = 0
    failures = []
    start = time.monotonic()
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers, initializer=init_worker) as pool:
        results = pool.imap_unordered(convert_one, tasks, chunksize=1)
        for completed, result in enumerate(results, 1):
            sample_id, status, message = result
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append((sample_id, message))

            if completed % args.report_every == 0 or completed == len(tasks):
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(tasks) - completed) / rate if rate else 0.0
                print(
                    f"Progress {completed}/{len(tasks)} "
                    f"({100.0 * completed / len(tasks):.2f}%); "
                    f"converted={converted}, skipped={skipped}, failed={len(failures)}; "
                    f"rate={rate:.2f} files/s; ETA={format_duration(eta)}",
                    flush=True,
                )

    elapsed = time.monotonic() - start
    print(
        f"Finished in {format_duration(elapsed)}: converted={converted}, "
        f"skipped={skipped}, failed={len(failures)}",
        flush=True,
    )
    if failures:
        failure_path = output_dir / "precompute_failures.csv"
        with open(failure_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "error"])
            writer.writerows(failures)
        raise RuntimeError(f"Precomputation failed for {len(failures)} files; see {failure_path}")


if __name__ == "__main__":
    main()
