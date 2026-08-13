#!/usr/bin/env python3
"""Materialize each Flickr sample's exact 00000003.jpg on fast storage."""

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path


FRAME_NAME = "00000003.jpg"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-view-dir")
    parser.add_argument("--lists-dir")
    parser.add_argument("--extracted-frames-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def index_frame3_sources(lists_dir, extracted_frames_dir, sample_ids):
    wanted = {sample_id.encode(): sample_id for sample_id in sample_ids}
    sources = {}
    list_paths = sorted(Path(lists_dir).glob("*_frames4_*.txt"))
    for list_path in list_paths:
        with list_path.open("rb", buffering=4 * 1024 * 1024) as handle:
            for line in handle:
                source_path = line.split(None, 1)[0]
                if not source_path.endswith(b"/" + FRAME_NAME.encode()):
                    continue
                _, marker, relative_path = source_path.partition(b"/frames/")
                if not marker:
                    continue
                video_dir = relative_path.rsplit(b"/", 1)[0]
                video_id = video_dir.rsplit(b"/", 1)[1]
                if video_id.endswith(b".mp4"):
                    video_id = video_id[:-4]
                sample_id = wanted.get(video_id)
                if sample_id is not None:
                    sources[sample_id] = str(
                        Path(extracted_frames_dir) / relative_path.decode()
                    )

    missing = [sample_id for sample_id in sample_ids if sample_id not in sources]
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing exact {FRAME_NAME} for {len(missing)}/{len(sample_ids)} IDs; "
            f"examples: {examples}"
        )
    return sources


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


def materialize_one(task):
    sample_id, frame3_source, output_path = task
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return sample_id, "skipped", ""

    temporary_path = None
    try:
        if not os.path.isfile(frame3_source):
            raise FileNotFoundError(frame3_source)

        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(output_path),
            prefix=f".{sample_id}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
        shutil.copyfile(frame3_source, temporary_path)
        os.replace(temporary_path, output_path)
        return sample_id, "copied", ""
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

    sample_ids = load_manifest_ids(args.manifest)
    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lists_dir or args.extracted_frames_dir:
        if not args.lists_dir or not args.extracted_frames_dir:
            raise ValueError(
                "--lists-dir and --extracted-frames-dir must be provided together"
            )
        frame3_sources = index_frame3_sources(
            args.lists_dir, args.extracted_frames_dir, sample_ids
        )
    elif args.frame_view_dir:
        frame_view_dir = Path(args.frame_view_dir)
        frame3_sources = {}
        for sample_id in sample_ids:
            view_path = frame_view_dir / f"{sample_id}.jpg"
            selected_source = Path(os.readlink(view_path))
            if not selected_source.is_absolute():
                selected_source = view_path.parent / selected_source
            frame3_sources[sample_id] = str(selected_source.parent / FRAME_NAME)
    else:
        raise ValueError(
            "Provide either --frame-view-dir or both --lists-dir and "
            "--extracted-frames-dir"
        )

    tasks = []
    for sample_id in sample_ids:
        tasks.append(
            (
                sample_id,
                frame3_sources[sample_id],
                str(output_dir / f"{sample_id}.jpg"),
            )
        )
    tasks.sort(key=lambda task: task[1])

    print(
        f"Frame-3 copies: {len(tasks)}, workers: {args.workers}, "
        f"output: {output_dir}",
        flush=True,
    )
    copied = 0
    skipped = 0
    failures = []
    start = time.monotonic()
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(materialize_one, tasks, chunksize=8)
        for completed, result in enumerate(results, 1):
            sample_id, status, message = result
            if status == "copied":
                copied += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append((sample_id, message))

            if completed % args.report_every == 0 or completed == len(tasks):
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(tasks) - completed) / rate if rate else 0.0
                print(
                    f"Frame progress {completed}/{len(tasks)} "
                    f"({100.0 * completed / len(tasks):.2f}%); "
                    f"copied={copied}, skipped={skipped}, failed={len(failures)}; "
                    f"rate={rate:.2f} files/s; ETA={format_duration(eta)}",
                    flush=True,
                )

    elapsed = time.monotonic() - start
    print(
        f"Frames finished in {format_duration(elapsed)}: copied={copied}, "
        f"skipped={skipped}, failed={len(failures)}",
        flush=True,
    )
    if failures:
        failure_path = output_dir / "frame_copy_failures.csv"
        with open(failure_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "error"])
            writer.writerows(failures)
        raise RuntimeError(
            f"Frame copy failed for {len(failures)} files; see {failure_path}"
        )


if __name__ == "__main__":
    main()
