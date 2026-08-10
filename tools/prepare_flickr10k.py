#!/usr/bin/env python3
"""Extract a Flickr-SoundNet subset from the official SoundNet archives.

The output is the flat layout expected by dataset.py:

    OUTPUT/
      audio/<video_id>.mp3
      frames/<video_id>.jpg
"""

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path


FRAME_NAMES = ("00000008.jpg", "00000013.jpg", "00000003.jpg", "00000018.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Text/CSV file containing video IDs")
    parser.add_argument("--lists-archive", required=True)
    parser.add_argument("--frames-archive", required=True)
    parser.add_argument("--audio-archive", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_ids(path):
    with open(path, encoding="utf-8") as file:
        return {line.strip().split(",")[0] for line in file if line.strip()}


def read_video_paths(archive_path, wanted_ids):
    result = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember("lists/train_videos.txt")
        with archive.extractfile(member) as file:
            for raw_line in file:
                video_path = raw_line.decode("utf-8").strip()
                video_id = Path(video_path).stem
                if video_id in wanted_ids:
                    result[video_id] = video_path
    return result


def copy_member(archive, member, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with archive.extractfile(member) as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(temporary, destination)


def extract_audio(archive_path, video_paths, output_dir):
    targets = {
        f"mp3/{video_path}.mp3": video_id
        for video_id, video_path in video_paths.items()
        if not (output_dir / f"{video_id}.mp3").is_file()
    }
    if not targets:
        return

    extracted = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            video_id = targets.pop(member.name, None)
            if video_id is None:
                continue
            copy_member(archive, member, output_dir / f"{video_id}.mp3")
            extracted += 1
            if extracted % 500 == 0:
                print(f"audio: extracted {extracted}, remaining {len(targets)}", flush=True)
            if not targets:
                break
    if targets:
        print(f"audio: {len(targets)} requested members were absent", flush=True)


def extract_frames(archive_path, video_paths, output_dir):
    target_members = {}
    for video_id, video_path in video_paths.items():
        for priority, frame_name in enumerate(FRAME_NAMES):
            target_members[f"frames/{video_path}/{frame_name}"] = (video_id, priority)

    selected_priority = {}
    completed = 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            target = target_members.get(member.name)
            if target is None:
                continue
            video_id, priority = target
            if priority >= selected_priority.get(video_id, len(FRAME_NAMES)):
                continue
            first_frame = video_id not in selected_priority
            copy_member(archive, member, output_dir / f"{video_id}.jpg")
            selected_priority[video_id] = priority
            if first_frame:
                completed += 1
                if completed % 500 == 0:
                    print(f"frames: found {completed}/{len(video_paths)} videos", flush=True)

    missing = set(video_paths).difference(selected_priority)
    if missing:
        print(f"frames: {len(missing)} videos had no candidate frame", flush=True)


def main():
    args = parse_args()
    wanted_ids = read_ids(args.manifest)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    frames_dir = output_dir / "frames"
    audio_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"manifest: {len(wanted_ids)} IDs", flush=True)
    video_paths = read_video_paths(args.lists_archive, wanted_ids)
    print(f"archive index: matched {len(video_paths)} IDs", flush=True)

    extract_audio(args.audio_archive, video_paths, audio_dir)
    extract_frames(args.frames_archive, video_paths, frames_dir)

    audio_ids = {path.stem for path in audio_dir.glob("*.mp3")}
    frame_ids = {path.stem for path in frames_dir.glob("*.jpg")}
    available_ids = sorted(wanted_ids.intersection(audio_ids, frame_ids))
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "requested": len(wanted_ids),
        "indexed": len(video_paths),
        "audio": len(wanted_ids.intersection(audio_ids)),
        "frames": len(wanted_ids.intersection(frame_ids)),
        "paired": len(available_ids),
        "frame_preference": list(FRAME_NAMES),
    }
    with open(output_dir / "preparation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    with open(output_dir / "available_ids.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(available_ids) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
