#!/usr/bin/env python3
"""Validate VGGSound train/test metadata and local media coverage."""

import argparse
import csv
import json
import re
from pathlib import Path


CLIP_SUFFIX = re.compile(r"_[0-9]{6}$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--test-data-root", required=True)
    parser.add_argument("--prepared-root")
    parser.add_argument(
        "--allow-source-overlap", action="store_true",
        help="Allow different timestamped clips from the same source video",
    )
    return parser.parse_args()


def read_ids(path):
    with open(path, newline="", encoding="utf-8") as file:
        ids = [row[0].strip() for row in csv.reader(file) if row and row[0].strip()]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {path}: {len(ids) - len(set(ids))}")
    return set(ids)


def media_ids(path, suffix):
    return {item.stem for item in Path(path).glob(f"*{suffix}") if item.is_file()}


def require_complete(name, wanted, available):
    missing = wanted.difference(available)
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"{name} is incomplete: missing {len(missing)}/{len(wanted)} IDs. "
            f"Examples: {examples}"
        )


def source_ids(ids):
    return {CLIP_SUFFIX.sub("", item) for item in ids}


def main():
    args = parse_args()
    train_ids = read_ids(args.train_manifest)
    test_ids = read_ids(args.test_manifest)

    exact_overlap = train_ids.intersection(test_ids)
    source_overlap = source_ids(train_ids).intersection(source_ids(test_ids))
    if exact_overlap:
        raise RuntimeError(
            f"Train/test clip leakage: {len(exact_overlap)} exact IDs; "
            f"examples: {', '.join(sorted(exact_overlap)[:5])}"
        )
    if source_overlap and not args.allow_source_overlap:
        raise RuntimeError(
            f"Train/test source-video leakage: {len(source_overlap)} YouTube IDs; "
            f"examples: {', '.join(sorted(source_overlap)[:5])}"
        )

    require_complete("Raw training videos", train_ids, media_ids(args.video_dir, ".mp4"))

    test_root = Path(args.test_data_root)
    require_complete("Test frames", test_ids, media_ids(test_root / "frames", ".jpg"))
    require_complete("Test audio", test_ids, media_ids(test_root / "audio", ".wav"))

    with open(args.annotations, encoding="utf-8") as file:
        annotations = json.load(file)
    annotation_ids = [item["file"] for item in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise RuntimeError("Duplicate IDs in test annotations")
    if set(annotation_ids) != test_ids:
        missing = test_ids.difference(annotation_ids)
        extra = set(annotation_ids).difference(test_ids)
        raise RuntimeError(
            f"Test annotations do not match test manifest: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    if args.prepared_root:
        prepared_root = Path(args.prepared_root)
        require_complete(
            "Prepared training frames", train_ids,
            media_ids(prepared_root / "frames", ".jpg")
        )
        require_complete(
            "Prepared training audio", train_ids,
            media_ids(prepared_root / "audio", ".wav").union(
                media_ids(prepared_root / "audio", ".npy")
            )
        )

    result = {
        "train_ids": len(train_ids),
        "test_ids": len(test_ids),
        "exact_overlap": 0,
        "source_video_overlap": len(source_overlap),
        "source_video_overlap_allowed": args.allow_source_overlap,
        "raw_training_videos": len(train_ids),
        "test_frames": len(test_ids),
        "test_audio": len(test_ids),
        "test_annotations": len(annotation_ids),
        "prepared_training_pairs": len(train_ids) if args.prepared_root else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
