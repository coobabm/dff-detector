"""Tight face crops for the video face-swap detector.

  uv run extract_tight.py

Reads data/videos/*.csv (hq_raw, hq_clip, public). For each video, takes FRAMES
evenly spaced frames, finds the largest face (YuNet), crops 1.3x the face box and
resizes to SIZE x SIZE. Unlike video_scan.py (loose 2.2x crop, face ~100 px), the
face fills most of the image, so fine swap traces survive.

Output: data/img380/<source>/<video>_<k>.jpg, manifest data/<--out> (default swap_frames.csv).

  uv run extract_tight.py                                   # face-swap detector data
  uv run extract_tight.py --lists aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv --out aivid_frames.csv
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "img380"
FRAMES = 16
SIZE = 380
SCALE = 1.3
_det = None


def detector():
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create(str(ROOT / "weights" / "yunet.onnx"), "", (320, 320), 0.8)
    return _det


def tight_crop(frame):
    """Largest face, 1.3x box, square, SIZE px. None if no face."""
    h, w = frame.shape[:2]
    sc = min(1.0, 960 / max(h, w))
    small = cv2.resize(frame, (int(w * sc), int(h * sc))) if sc < 1 else frame
    d = detector()
    d.setInputSize((small.shape[1], small.shape[0]))
    _, dets = d.detect(small)
    if dets is None:
        return None
    x, y, bw, bh = dets[np.argmax(dets[:, 2] * dets[:, 3])][:4] / sc
    if min(bw, bh) < 40:
        return None
    s = max(bw, bh) * SCALE
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def video_frames(path, n=FRAMES):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for idx in sorted({int(total * (i + 0.5) / n) for i in range(n)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out.append((idx, frame))
    return out


def process(row):
    vid = Path(row["path"]).stem
    rows = []
    for k, (_, frame) in enumerate(video_frames(DATA / row["path"])):
        crop = tight_crop(frame)
        if crop is None:
            continue
        dst = OUT / row["source"] / f"{vid}_{k}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([str(dst.relative_to(DATA)), row["source"], row["label"], row["group"], row["split"],
                     f"{row['source']}/{vid}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=None,
                    help="video manifests under data/videos (default: all *.csv there)")
    ap.add_argument("--out", default="swap_frames.csv", help="frame manifest written under data/")
    args = ap.parse_args()
    lists = [DATA / "videos" / l for l in args.lists] if args.lists else sorted((DATA / "videos").glob("*.csv"))
    videos = []
    for f in lists:
        videos += list(csv.DictReader(open(f)))
    print("videos", len(videos), flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(process, videos, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(f"{i}/{len(videos)} videos, {len(rows)} frames", flush=True)
    with open(DATA / args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "group", "split", "video"])
        w.writerows(rows)
    print("frames", len(rows))


if __name__ == "__main__":
    main()
