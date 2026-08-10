#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 ARCHIVE_DIR OUTPUT_DIR"
    exit 2
fi

archive_dir=$1
output_dir=$2
mkdir -p "$output_dir"

extract_archive() {
    archive_name=$1
    marker="$output_dir/.${archive_name}.complete"
    if [ -f "$marker" ]; then
        echo "skip completed archive: $archive_name"
        return
    fi

    echo "start: $archive_name"
    tar --use-compress-program=pigz \
        --checkpoint=100000 \
        --checkpoint-action="echo=$archive_name: %u files processed" \
        -xf "$archive_dir/$archive_name" -C "$output_dir"
    touch "$marker"
    echo "complete: $archive_name"
}

extract_archive lists_public.tar.gz
extract_archive frames_public.tar.gz
extract_archive mp3_public.tar.gz
