#!/usr/bin/env bash
#
# Create a short clip from a video in the pipeline data directory.
#
# Usage:
#   ./scripts/create_clip.sh <source_stem> <clip_stem> [start] [duration]
#
# Arguments:
#   source_stem  — filename stem of the source video (without .mp4)
#   clip_stem    — filename stem for the output clip
#   start        — start time (default: 00:04:00)
#   duration     — clip length (default: 00:05:00)
#
# Examples:
#   # Default 5-min clip starting at 4:00
#   ./scripts/create_clip.sh \
#     "Warriors & Lakers Instant Classic - 2021 Play-In Tournament" \
#     "Warriors vs Lakers Q1 Clip"
#
#   # Custom: 3 minutes starting at 10:30
#   ./scripts/create_clip.sh \
#     "Warriors & Lakers Instant Classic - 2021 Play-In Tournament" \
#     "Warriors vs Lakers Q2 Clip" \
#     00:10:30 00:03:00

set -euo pipefail

VIDEOS_DIR="pipeline_data/api/videos"

SOURCE_STEM="${1:?Usage: $0 <source_stem> <clip_stem> [start] [duration]}"
CLIP_STEM="${2:?Usage: $0 <source_stem> <clip_stem> [start] [duration]}"
START="${3:-00:04:00}"
DURATION="${4:-00:05:00}"

SOURCE="${VIDEOS_DIR}/${SOURCE_STEM}.mp4"
OUTPUT="${VIDEOS_DIR}/${CLIP_STEM}.mp4"

if [ ! -f "$SOURCE" ]; then
    echo "Error: source video not found: $SOURCE"
    exit 1
fi

if [ -f "$OUTPUT" ]; then
    echo "Clip already exists: $OUTPUT"
    exit 0
fi

echo "Creating clip:"
echo "  Source:   $SOURCE"
echo "  Output:   $OUTPUT"
echo "  Start:    $START"
echo "  Duration: $DURATION"

ffmpeg -y -loglevel error \
    -ss "$START" \
    -i "$SOURCE" \
    -t "$DURATION" \
    -c:v libx264 -crf 23 \
    -c:a aac \
    "$OUTPUT"

echo "Done. Clip saved to: $OUTPUT"
echo ""
echo "Add to video_registry.yml:"
echo ""
echo "  - id: <unique-id>"
echo "    title: \"${CLIP_STEM}\""
echo "    url: local"
echo "    source_language: en"
