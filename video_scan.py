"""Scan videos: sample frames, detect faces, score each face with every trained model.

  uv run video_scan.py video1.mp4 video2.mp4 ... [--fps 2]

Prints per-video summary; writes a contact sheet of face crops with scores
to scans/<video>.jpg and raw scores to scans/<video>.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from train import build_model, eval_tf

ROOT = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
YUNET = ROOT / "weights" / "yunet.onnx"
if not YUNET.exists():
    import urllib.request
    YUNET.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                               "face_detection_yunet_2023mar.onnx", YUNET)
DETECTOR = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8)


def load_models():
    models = {}
    for run in sorted((ROOT / "runs").iterdir()):
        if (run / "model.pt").exists():
            m = build_model(pretrained=False)
            m.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
            models[run.name] = m.to(DEVICE).eval()
    return models


def faces_in(frame_bgr):
    h, w = frame_bgr.shape[:2]
    DETECTOR.setInputSize((w, h))
    _, dets = DETECTOR.detect(frame_bgr)
    out = []
    for det in dets if dets is not None else []:
        x, y, bw, bh = (int(v) for v in det[:4])
        if min(bw, bh) < max(24, w // 40):
            continue
        # DFF images are loose face crops: face ~45% of width, some shoulders below.
        s = int(max(bw, bh) * 2.2)
        cx, cy = x + bw // 2, y + int(bh * 0.6)
        x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
        x1, y1 = min(w, x0 + s), min(h, y0 + s)
        crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        out.append(((int(x), int(y), int(bw), int(bh)), Image.fromarray(crop).resize((256, 256), Image.BICUBIC)))
    return out


@torch.no_grad()
def scan(path, models, fps):
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(vfps / fps))
    rows, crops, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            H = frame.shape[0]
            for box, crop in faces_in(frame):
                x = eval_tf(crop).unsqueeze(0).to(DEVICE)
                scores = {k: float(torch.sigmoid(m(x)).item()) for k, m in models.items()}
                rows.append({"t": round(i / vfps, 2), "box": box, "half": "top" if box[1] + box[3] / 2 < H / 2 else "bottom", **scores})
                crops.append(crop)
        i += 1
    return rows, crops


def sheet(rows, crops, key, dst, n=24):
    if not rows:
        return
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    cols = 8
    im = Image.new("RGB", (cols * 160, ((len(idx) + cols - 1) // cols) * 180), "white")
    d = ImageDraw.Draw(im)
    for k, j in enumerate(idx):
        x, y = (k % cols) * 160, (k // cols) * 180
        im.paste(crops[j].resize((160, 160)), (x, y))
        p = rows[j][key]
        d.text((x + 4, y + 162), f"t={rows[j]['t']}s {key}={p:.2f}", fill="red" if p > 0.5 else "green")
    im.save(dst, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=2)
    args = ap.parse_args()
    models = load_models()
    out = ROOT / "scans"
    out.mkdir(exist_ok=True)
    for v in args.videos:
        v = Path(v)
        rows, crops = scan(v, models, args.fps)
        json.dump(rows, open(out / f"{v.stem}.json", "w"), ensure_ascii=False)
        sheet(rows, crops, "all" if "all" in models else list(models)[0], out / f"{v.stem}.jpg")
        print(f"\n{v.name}: {len(rows)} faces")
        if not rows:
            continue
        halves = sorted({r["half"] for r in rows})
        for half in halves if len(halves) > 1 else [None]:
            sub = [r for r in rows if half is None or r["half"] == half]
            label = f"  [{half}] {len(sub)} faces" if half else ""
            if label:
                print(label)
            for k in models:
                ps = np.array([r[k] for r in sub])
                print(f"    {k:8s} median {np.median(ps):.2f}  frames>0.5 {np.mean(ps > 0.5):.0%}")


if __name__ == "__main__":
    main()
