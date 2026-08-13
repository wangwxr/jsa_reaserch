#!/usr/bin/env python3
"""Extract the middle frame and middle five seconds from VGGSound MP4 files."""

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def read_ids(path):
    with open(path, newline="", encoding="utf-8") as file:
        ids = [row[0].strip() for row in csv.reader(file) if row and row[0].strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {path}: {len(ids) - len(set(ids))}")
    return sorted(ids)


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)

#这个最重要
def prepare_one(video_id, video_dir, audio_dir, frame_dir):
    source = video_dir / f"{video_id}.mp4"
    audio_path = audio_dir / f"{video_id}.wav"
    frame_path = frame_dir / f"{video_id}.jpg"
    if audio_path.is_file() and frame_path.is_file():
        return video_id, "cached", ""
    if not source.is_file():
        return video_id, "failed", "source MP4 is missing"

    audio_tmp = audio_dir / f"{video_id}.tmp.wav"
    frame_tmp = frame_dir / f"{video_id}.tmp.jpg"
    try:
        probe = run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(source),
        ])
        duration = float(probe.stdout.strip())
        audio_start = max(0.0, (duration - 5.0) / 2.0) #取整个视频的middle frame
        frame_offset = min(2.5, duration / 2.0) #指的是相对audio开始向后走2.5

        run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{audio_start:.6f}", "-i", str(source),
            "-map", "0:a:0", "-t", "5", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(audio_tmp),
            "-map", "0:v:0", "-ss", f"{frame_offset:.6f}",
            "-frames:v", "1", "-q:v", "2", str(frame_tmp),
        ])
        os.replace(audio_tmp, audio_path)
        os.replace(frame_tmp, frame_path)
        return video_id, "prepared", ""
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        audio_tmp.unlink(missing_ok=True)
        frame_tmp.unlink(missing_ok=True)
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        return video_id, "failed", detail


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    video_ids = read_ids(args.manifest)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    frame_dir = output_dir / "frames"
    audio_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    counts = {"prepared": 0, "cached": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_one, video_id, video_dir, audio_dir, frame_dir): video_id
            for video_id in video_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            video_id, status, detail = future.result()
            counts[status] += 1
            if status == "failed":
                failures.append({"id": video_id, "error": detail})
            if completed % 100 == 0 or completed == len(video_ids):
                print(
                    f"processed {completed}/{len(video_ids)} "
                    f"(new={counts['prepared']}, cached={counts['cached']}, "
                    f"failed={counts['failed']})",
                    flush=True,
                )

    available = sorted(
        {path.stem for path in audio_dir.glob("*.wav")}.intersection(
            path.stem for path in frame_dir.glob("*.jpg")
        ).intersection(video_ids)
    )
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "requested": len(video_ids),
        "available": len(available),
        **counts,
        "failures": failures,
    }
    with open(output_dir / "available_ids.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(available) + "\n")
    with open(output_dir / "preparation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    if failures or len(available) != len(video_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
