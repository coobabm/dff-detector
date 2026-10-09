"""Fine-tune the image detector on face frames from AI / real videos.

  uv run video_finetune.py

Starts from runs/all_openai. Train mix per epoch (balanced real/fake):
  real: real video frames (FF++, CelebV-HQ clips, CelebV-HQ raw) + the 6 real photo sources
  fake: AI video frames (face swap, talking head, text-to-video) + DFF fakes + OpenAI images
Video frames are split by video, never by frame. Reports frame- and video-level AUC
per source on held-out videos, and checks the old image tests did not regress.
Saves runs/all_video/model.pt and results/video_finetune.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from openai_curve import load_sets
from train import DATA, Faces, build_model, predict, train_tf

ROOT = Path(__file__).parent
EPOCHS = 3
PER_CLASS = 8000


def load_video():
    parts = [pd.read_csv(f) for f in sorted((DATA / "video_manifest").glob("*.csv"))]
    return pd.concat(parts, ignore_index=True)


def video_level(df, p):
    d = df.assign(p=p).groupby(["video", "source", "label"], as_index=False).p.mean()
    return d


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def evaluate(model, vid_te, sets, device):
    res = {}
    p = predict(model, vid_te, device)
    v = video_level(vid_te, p)
    real_f, real_v = p[vid_te.label.values == 0], v[v.label == 0].p.values
    res["real_video_frame_acc"] = float((real_f < 0.5).mean())
    res["real_video_acc"] = float((real_v < 0.5).mean())
    res["real_video_acc_by_source"] = {k: float((g.p < 0.5).mean()) for k, g in v[v.label == 0].groupby("source")}
    for src, g in v[v.label == 1].groupby("source"):
        fp = p[(vid_te.source == src).values]
        res[src] = {
            "videos": len(g),
            "frame_auc": auc([0] * len(real_f) + [1] * len(fp), list(real_f) + list(fp)),
            "video_auc": auc([0] * len(real_v) + [1] * len(g), list(real_v) + list(g.p)),
            "video_recall": float((g.p > 0.5).mean()),
        }
    # Regression check on the earlier image tests.
    po, pr, pd_ = (predict(model, sets[k], device) for k in ("oa_test", "real_test", "dff_fake_test"))
    res["openai_recall"] = float((po > 0.5).mean())
    res["photo_real_acc"] = float((pr < 0.5).mean())
    res["dff_fake_recall"] = float((pd_ > 0.5).mean())
    return res


def main():
    os.environ["DFF_CINEMATIC"] = "1"  # film-look augmentation, inherited by DataLoader workers
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    sets = load_sets()
    vid = load_video()
    vid_tr, vid_te = vid[vid.split == "train"], vid[vid.split == "test"]
    print("video frames", vid.groupby(["source", "split"]).size().unstack(fill_value=0).to_string(), flush=True)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "runs" / "all_openai" / "model.pt", map_location="cpu"))
    model.to(device)
    before = evaluate(model, vid_te, sets, device)
    print("BEFORE", json.dumps(before), flush=True)

    vr, vf = vid_tr[vid_tr.label == 0], vid_tr[vid_tr.label == 1]
    # Each fake source gets an equal share so big sources do not dominate.
    fake_parts = [g.sample(PER_CLASS // 2 // vf.source.nunique(), replace=True, random_state=0)
                  for _, g in vf.groupby("source")]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        # Real video frames: equal share per real source (FF++, CelebV clips, CelebV raw).
        real = pd.concat([*(g.sample(PER_CLASS // 2 // vr.source.nunique(), replace=True, random_state=ep)
                            for _, g in vr.groupby("source")),
                          sets["real_train"].sample(PER_CLASS // 2, replace=True, random_state=ep)])
        fake = pd.concat([*fake_parts,
                          sets["dff_fake_train"].sample(PER_CLASS // 4, random_state=ep),
                          sets["oa_pool"].sample(PER_CLASS // 4, replace=True, random_state=ep)])
        dl = DataLoader(Faces(pd.concat([real, fake]), train_tf), batch_size=64, shuffle=True,
                        num_workers=8, drop_last=True)
        model.train()
        tot = 0.0
        for i, (x, y) in enumerate(dl):
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            if i % 50 == 0:
                print(f"ep{ep} {i}/{len(dl)} loss {tot / (i + 1):.4f}", flush=True)

    after = evaluate(model, vid_te, sets, device)
    print("AFTER", json.dumps(after), flush=True)
    out = ROOT / "runs" / "all_video"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"before": before, "after": after}, open(ROOT / "results" / "video_finetune.json", "w"), indent=1)


if __name__ == "__main__":
    main()
