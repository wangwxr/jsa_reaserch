#!/usr/bin/env python3
"""Build a flat Flickr-SoundNet subset using symlinks to a full extraction."""

import argparse
import json
from pathlib import Path


FRAME_NAMES = ("00000008.jpg", "00000013.jpg", "00000003.jpg", "00000018.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--extracted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_ids(path):
    with open(path, encoding="utf-8") as file:
        return {line.strip().split(",")[0] for line in file if line.strip()}


def load_video_paths(path, wanted_ids):
    result = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            video_path = line.strip()
            video_id = Path(video_path).stem
            if video_id in wanted_ids:
                result[video_id] = video_path
    return result


def load_frame_paths(lists_dir, wanted_ids):
    """Index the preferred available frame from SoundNet's frame manifests."""
    result = {}
    ranks = {name: rank for rank, name in enumerate(FRAME_NAMES)}
    for list_path in sorted(lists_dir.glob("*_frames4_*.txt")):
        with open(list_path, encoding="utf-8") as file:
            for line in file:
                source_path = line.split(maxsplit=1)[0]
                _, marker, relative_path = source_path.partition("/frames/")
                if not marker:
                    continue
                frame_root, frame_name = relative_path.rsplit("/", 1)
                video_id = Path(frame_root).stem
                if video_id not in wanted_ids or frame_name not in ranks:
                    continue
                current = result.get(video_id)
                if current is None or ranks[frame_name] < current[0]:
                    result[video_id] = (ranks[frame_name], relative_path)
    return {video_id: value[1] for video_id, value in result.items()}


def create_symlink(source, destination):
    source = source.absolute()
    if destination.is_symlink():
        target = destination.readlink()
        if not target.is_absolute():
            target = (destination.parent / target).absolute()
        if target == source:
            return
        raise RuntimeError(f"existing symlink points elsewhere: {destination}")
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source)


def main():
    args = parse_args()
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    frame_dir = output_dir / "frames"
    audio_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    wanted_ids = load_ids(args.manifest)
    lists_dir = extracted_dir / "lists"
    video_paths = {}
    for split in ("train", "val"):
        video_paths.update(
            load_video_paths(lists_dir / f"{split}_videos.txt", wanted_ids)
        )
    complete_extraction = all(
        (extracted_dir / f".{archive}.complete").is_file()
        for archive in ("frames_public.tar.gz", "mp3_public.tar.gz")
    )
    frame_paths = (
        load_frame_paths(lists_dir, wanted_ids) if complete_extraction else {}
    )

    audio_count = 0
    frame_count = 0
    paired_ids = []
    for index, video_id in enumerate(sorted(video_paths), start=1):
        video_path = video_paths[video_id]
        audio_source = extracted_dir / "mp3" / f"{video_path}.mp3"
        if complete_extraction:
            frame_relative = frame_paths.get(video_id)
            frame_source = (
                extracted_dir / "frames" / frame_relative
                if frame_relative is not None else None
            )
            audio_available = audio_source.is_file()
            if frame_source is not None and not frame_source.is_file():
                frame_source = None
        else:
            frame_root = extracted_dir / "frames" / video_path
            frame_source = next(
                (frame_root / name for name in FRAME_NAMES
                 if (frame_root / name).is_file()),
                None,
            )
            audio_available = audio_source.is_file()

        if audio_available:
            create_symlink(audio_source, audio_dir / f"{video_id}.mp3")
            audio_count += 1
        if frame_source is not None:
            create_symlink(frame_source, frame_dir / f"{video_id}.jpg")
            frame_count += 1
        if audio_available and frame_source is not None:
            paired_ids.append(video_id)

        if index % 10000 == 0:
            print(f"processed {index}/{len(video_paths)} indexed IDs", flush=True)

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "requested": len(wanted_ids),
        "indexed": len(video_paths),
        "audio": audio_count,
        "frames": frame_count,
        "paired": len(paired_ids),
        "frame_preference": list(FRAME_NAMES),
        "trusted_complete_extraction": complete_extraction,
    }
    with open(output_dir / "available_ids.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(paired_ids) + "\n")
    with open(output_dir / "preparation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
