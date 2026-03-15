"""
Convert privileged video pairs to side-by-side comparison GIFs.

Usage:
    # Convert specific pair by ID
    python create_comparison_gif.py --id 1

    # Convert multiple pairs
    python create_comparison_gif.py --id 1 2 3

    # Convert all pairs
    python create_comparison_gif.py --all

    # Output for GitHub Pages (docs/static/videos/)
    python create_comparison_gif.py --id 1 2 3 --output_dir docs/static/videos
"""

import argparse
import os
import sys
import csv
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # VideoFPS/
PAIRS_DIR = os.path.join(REPO_ROOT, "privileged_video_pairs")
MANIFEST = os.path.join(PAIRS_DIR, "manifest.csv")


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_video_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=10,
    )
    return float(result.stdout.strip())


def create_comparison_gif(pair, output_dir, max_duration=6.0, fps=10, width=480):
    """Create a side-by-side comparison GIF from a video pair."""
    pair_id = int(pair["id"])
    original = os.path.join(PAIRS_DIR, pair["original_file"])
    corrected = os.path.join(PAIRS_DIR, pair["corrected_file"])

    if not os.path.exists(original) or not os.path.exists(corrected):
        print(f"  [SKIP] Missing files for pair {pair_id}")
        return None

    meta_fps = pair.get("meta_fps", "?")
    pred_fps = pair.get("predicted_fps", "?")
    prompt = pair.get("prompt_text", "")
    model = pair.get("model", "")
    source = pair.get("source_video", "")

    # Truncate prompt for title
    title = prompt[:70] + "..." if len(prompt) > 70 else prompt
    if not title:
        title = source

    # Determine duration (use shorter of the two, capped)
    dur_orig = get_video_duration(original)
    dur_corr = get_video_duration(corrected)
    duration = min(dur_orig, dur_corr, max_duration)

    # Format FPS labels
    try:
        meta_label = f"Meta FPS: {float(meta_fps):.0f}"
    except (ValueError, TypeError):
        meta_label = f"Meta FPS: {meta_fps}"
    try:
        pred_label = f"PhyFPS: {float(pred_fps):.1f}"
    except (ValueError, TypeError):
        pred_label = f"PhyFPS: {pred_fps}"

    out_name = f"pair_{pair_id:02d}.gif"
    out_path = os.path.join(output_dir, out_name)

    # Build ffmpeg filter for side-by-side with labels
    filter_complex = (
        f"[0:v]trim=0:{duration},setpts=PTS-STARTPTS,scale={width}:-2,fps={fps}[left];"
        f"[1:v]trim=0:{duration},setpts=PTS-STARTPTS,scale={width}:-2,fps={fps}[right];"
        f"[left]drawtext=text='Original ({meta_label})':fontsize=18:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-30[left_t];"
        f"[right]drawtext=text='Corrected ({pred_label})':fontsize=18:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-30[right_t];"
        f"[left_t][right_t]hstack=inputs=2[stacked];"
        f"[stacked]split[s0][s1];"
        f"[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", original,
        "-i", corrected,
        "-filter_complex", filter_complex,
        "-t", str(duration),
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  [ERROR] Pair {pair_id}: {result.stderr[-200:]}")
        return None

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  [OK] Pair {pair_id}: {out_name} ({size_mb:.1f} MB) — {title[:50]}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison GIFs")
    parser.add_argument("--id", type=int, nargs="+", help="Pair ID(s) to convert")
    parser.add_argument("--all", action="store_true", help="Convert all pairs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: privileged_video_pairs/gifs/)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    parser.add_argument("--width", type=int, default=480, help="Width per side")
    parser.add_argument("--max_duration", type=float, default=6.0, help="Max duration in seconds")
    args = parser.parse_args()

    if not args.id and not args.all:
        parser.error("Provide --id or --all")

    manifest = load_manifest()
    output_dir = args.output_dir or os.path.join(PAIRS_DIR, "gifs")
    os.makedirs(output_dir, exist_ok=True)

    if args.all:
        pairs = manifest
    else:
        id_set = set(args.id)
        pairs = [p for p in manifest if int(p["id"]) in id_set]

    print(f"Converting {len(pairs)} pair(s) to GIFs in {output_dir}/\n")
    created = []
    for pair in pairs:
        result = create_comparison_gif(pair, output_dir, args.max_duration, args.fps, args.width)
        if result:
            created.append(result)

    print(f"\nDone. Created {len(created)}/{len(pairs)} GIFs.")


if __name__ == "__main__":
    main()
