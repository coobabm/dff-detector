"""Compare single AI-video detectors (v1-v4) with their ensemble.

  uv run ensemble_eval.py

Score per video = median over frames. Ensemble = mean over models of
(logit(p) - logit(threshold)), i.e. each model's margin above its own threshold;
ensemble flags a video when the mean margin > 0. Evaluated on the held-out test
videos (aivid_frames.csv, test split) and on the local example videos.
Writes results/aivid_ensemble.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from extract_tight import tight_crop, video_frames
from train_swap import DATA, DEVICE, build, eval_tf, predict

ROOT = Path(__file__).parent
VERSIONS = ["v1", "v2", "v3", "v4"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load(v):
    ck = torch.load(ROOT / "runs_aivid" / v / "model.pt", map_location="cpu")
    thr = json.load(open(ROOT / "results" / f"aivid_detector_{v}.json"))["test"]["threshold"]
    m = build(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(DEVICE).eval(), thr


def main():
    df = pd.read_csv(DATA / "aivid_frames.csv")
    te = df[df.split == "test"]
    te = te[te.source.str.endswith("_real") | te.source.str.contains(
        r"^rp_|^(veo3|sora2|kling|wan2|sora1|keling)$|^thb_|^df40_heygen", regex=True)]
    vids = sys.argv[1:]
    crops = {}
    for v in vids:
        crops[v] = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                    for _, f in video_frames(v, 32) if (c := tight_crop(f)) is not None]
    margins, local, res = {}, {}, {}
    for ver in VERSIONS:
        m, thr = load(ver)
        g = te.assign(p=predict(m, te)).groupby(["video", "label"], as_index=False).p.median()
        margins[ver] = logit(g.p.values) - logit(thr)
        res[ver] = {"auc": float(roc_auc_score(g.label, g.p)),
                    "real_acc": float((g[g.label == 0].p <= thr).mean()),
                    "fake_recall": float((g[g.label == 1].p > thr).mean())}
        with torch.no_grad():
            for v in vids:
                ps = [torch.sigmoid(m(eval_tf(c).unsqueeze(0).to(DEVICE))).item() for c in crops[v]]
                local.setdefault(Path(v).name, {})[ver] = float(logit(np.median(ps)) - logit(thr))
        labels = g.label.values
        print(ver, res[ver], flush=True)
    ens = np.mean([margins[v] for v in VERSIONS], axis=0)
    res["ensemble"] = {"auc": float(roc_auc_score(labels, ens)), "real_acc": float((ens[labels == 0] <= 0).mean()),
                       "fake_recall": float((ens[labels == 1] > 0).mean())}
    print("ensemble", res["ensemble"])
    for name, d in local.items():
        d["ensemble"] = float(np.mean([d[v] for v in VERSIONS]))
        print(f"{name:45s} " + "  ".join(f"{k} {'FAKE' if s > 0 else 'real'}({s:+.1f})" for k, s in d.items()))
    json.dump({"test": res, "local_margins": local}, open(ROOT / "results" / "aivid_ensemble.json", "w"), indent=1)


if __name__ == "__main__":
    main()
