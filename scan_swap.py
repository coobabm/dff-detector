"""Scan videos with the face-swap detector (tight 380 px face crops).

  uv run scan_swap.py video1.mp4 video2.mp4 ...                  # face-swap detector
  uv run scan_swap.py --model runs_aivid/b4 video1.mp4 ...       # AI-video detector

Per video: 32 evenly spaced frames, largest face, median fake score vs. the
threshold chosen on validation. Also writes a contact sheet to scans_swap/<video>.jpg.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from extract_tight import tight_crop, video_frames
from train_swap import DEVICE, OUT, build, eval_tf

ROOT = Path(__file__).parent


def main():
    args = sys.argv[1:]
    model_dir = OUT
    if args and args[0] == "--model":
        model_dir, args = ROOT / args[1], args[2:]
    ck = torch.load(model_dir / "model.pt", map_location="cpu")
    model = build(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    thr = ck["threshold"]
    (ROOT / "scans_swap").mkdir(exist_ok=True)
    print(f"threshold {thr:.2f}")
    for v in args:
        crops, ps = [], []
        for _, frame in video_frames(v, 32):
            c = tight_crop(frame)
            if c is None:
                continue
            im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                ps.append(torch.sigmoid(model(eval_tf(im).unsqueeze(0).to(DEVICE))).item())
            crops.append(im)
        if not ps:
            print(f"{Path(v).name}: no face")
            continue
        med = float(np.median(ps))
        verdict = "FAKE" if med > thr else "real"
        print(f"{Path(v).name}: {len(ps)} faces  median {med:.2f}  frames>thr {np.mean(np.array(ps) > thr):.0%}  -> {verdict}")
        sheet = Image.new("RGB", (8 * 160, ((len(crops) + 7) // 8) * 180), "white")
        d = ImageDraw.Draw(sheet)
        for k, (im, p) in enumerate(zip(crops, ps)):
            x, y = (k % 8) * 160, (k // 8) * 180
            sheet.paste(im.resize((160, 160)), (x, y))
            d.text((x + 4, y + 162), f"{p:.2f}", fill="red" if p > thr else "green")
        sheet.save(ROOT / "scans_swap" / f"{Path(v).stem}.jpg", quality=88)


if __name__ == "__main__":
    main()
