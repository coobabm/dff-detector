"""Make "web re-upload" copies of training videos: downscale + strong H.264 / VP9.

  uv run recompress_videos.py aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv

Videos shared online (YouTube, Wikimedia, messaging apps) are re-encoded at lower
resolution and bitrate, which wipes out fine generator traces. Detectors trained only
on pristine files then miss them. Every real and fake video gets one copy with random
settings (height 360/480/720; libx264 crf 26-34 or VP9 crf 36-46), same label, group
and split as the original.
Output: data/videos/recomp/<source>/<name>__rc.mp4, manifest data/videos/recomp.csv.
"""
import csv
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA = Path(__file__).parent / "data"


def recompress(row):
    src = DATA / row["path"]
    rng = random.Random(row["path"])
    h = rng.choice([360, 480, 480, 720])
    dst = DATA / "videos" / "recomp" / row["source"] / f"{src.stem}__rc.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        scale = f"scale=-2:'min({h},ih)'"
        if rng.random() < 0.3:
            # VP9 round trip (YouTube-like), then back to H.264 so OpenCV reads it everywhere.
            tmp = dst.with_suffix(".webm")
            cmds = [["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", scale, "-c:v", "libvpx-vp9",
                     "-crf", str(rng.randint(36, 46)), "-b:v", "0", "-an", str(tmp)],
                    ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18", "-an", str(dst)]]
        else:
            tmp = None
            cmds = [["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", scale, "-c:v", "libx264",
                     "-preset", "fast", "-crf", str(rng.randint(26, 34)), "-an", str(dst)]]
        for c in cmds:
            if subprocess.run(c, capture_output=True).returncode != 0:
                return None
        if tmp:
            tmp.unlink(missing_ok=True)
    return {**row, "path": str(dst.relative_to(DATA))}


def main():
    rows = []
    for name in sys.argv[1:]:
        rows += list(csv.DictReader(open(DATA / "videos" / name)))
    print("videos", len(rows), flush=True)
    out = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(recompress, rows, chunksize=4)):
            if r:
                out.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    with open(DATA / "videos" / "recomp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "label", "group", "split"])
        w.writeheader()
        w.writerows(out)
    print("done", len(out))


if __name__ == "__main__":
    main()
