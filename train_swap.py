"""Video face-swap detector: EfficientNet-B4 on tight 380 px face crops.

  uv run train_swap.py [--epochs 5]                                   # face-swap detector
  uv run train_swap.py --frames aivid_frames.csv --task aivid --out runs_aivid/b4   # AI-video detector

Data: data/swap_frames.csv from extract_tight.py. 10% of training groups are held
out as validation (best epoch + decision threshold chosen on video-level val scores).
Every split is by identity group (a real clip and its swaps share a group).
Reports frame/video AUC per source on the test split.
Saves runs_swap/b4/model.pt (+ threshold) and results/swap_detector.json.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from train import MEAN, STD, degrade

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "runs_swap" / "b4"  # default output (face-swap detector)
# Which fake sources each task trains on; reals are every *_real source.
TASK_FAKES = {
    "swap": r"^(hq_clip|hq_raw)_(?!real)|^ffpp_(?!real)|^df40_(inswap|simswap|uniface)",
    "aivid": r"^rp_|^(veo3|sora2|kling|wan2|sora1|keling)$|^thb_|^df40_heygen",
}
SIZE = 380
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train_tf = T.Compose([
    T.Lambda(degrade),
    T.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.15, 0.15, 0.15),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


class Frames(Dataset):
    def __init__(self, df, tf):
        self.p, self.y, self.tf = df.path.tolist(), df.label.tolist(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(DATA / self.p[i]).convert("RGB")), torch.tensor(float(self.y[i]))


def build(pretrained=True):
    return timm.create_model("efficientnet_b4.ra2_in1k", pretrained=pretrained, num_classes=1, drop_rate=0.3)


@torch.no_grad()
def predict(model, df):
    model.eval()
    out = []
    for x, _ in DataLoader(Frames(df, eval_tf), batch_size=32, num_workers=8):
        out.append(torch.sigmoid(model(x.to(DEVICE)).squeeze(1)).float().cpu())
    return torch.cat(out).numpy()


def video_scores(df, p):
    return df.assign(p=p).groupby(["video", "source", "label"], as_index=False).p.median()


def best_threshold(v):
    ts = np.linspace(0.05, 0.95, 91)
    acc = [((v.p > t) == (v.label == 1)).mean() for t in ts]
    return float(ts[int(np.argmax(acc))])


def report(df, p, thr):
    v = video_scores(df, p)
    real = v[v.label == 0]
    res = {"threshold": thr, "real_video_acc": float((real.p <= thr).mean()),
           "real_video_acc_by_source": {k: float((g.p <= thr).mean()) for k, g in real.groupby("source")}}
    for src, g in v[v.label == 1].groupby("source"):
        y = [0] * len(real) + [1] * len(g)
        res[src] = {"videos": len(g), "video_auc": float(roc_auc_score(y, list(real.p) + list(g.p))),
                    "video_recall": float((g.p > thr).mean())}
    y = [0] * len(real) + [1] * int((v.label == 1).sum())
    res["all_fakes_video_auc"] = float(roc_auc_score(y, list(real.p) + list(v[v.label == 1].p)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1500, help="optimizer steps per epoch")
    ap.add_argument("--frames", default="swap_frames.csv")
    ap.add_argument("--task", choices=list(TASK_FAKES), default="swap")
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)))
    args = ap.parse_args()
    out = ROOT / args.out
    os.environ["DFF_CINEMATIC"] = "1"
    torch.manual_seed(0)
    random.seed(0)

    df = pd.read_csv(DATA / args.frames)
    is_real = df.source.str.endswith("_real") & (df.label == 0)
    df = df[is_real | df.source.str.contains(TASK_FAKES[args.task], regex=True)]
    tr, te = df[df.split == "train"], df[df.split == "test"]
    groups = sorted(tr.group.unique())
    random.Random(0).shuffle(groups)
    val_groups = set(groups[: len(groups) // 10])
    va, tr = tr[tr.group.isin(val_groups)], tr[~tr.group.isin(val_groups)]
    print(df.groupby(["source", "split"]).video.nunique().unstack(fill_value=0).to_string(), flush=True)
    print("frames train/val/test", len(tr), len(va), len(te), flush=True)

    # Balance: half real, half fake; within each class every source gets an equal share.
    w = np.zeros(len(tr))
    for (lab, src), idx in tr.groupby(["label", "source"]).groups.items():
        n_src = tr[tr.label == lab].source.nunique()
        w[tr.index.get_indexer(idx)] = 1.0 / (2 * n_src * len(idx))
    sampler = WeightedRandomSampler(w, num_samples=args.steps * args.bs, replacement=True)

    model = build().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        dl = DataLoader(Frames(tr.reset_index(drop=True), train_tf), batch_size=args.bs, sampler=sampler,
                        num_workers=8, drop_last=True)
        tot = 0.0
        for i, (x, y) in enumerate(dl):
            loss = loss_fn(model(x.to(DEVICE)).squeeze(1), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"ep{ep} {i}/{args.steps} loss {tot / (i + 1):.4f}", flush=True)
        v = video_scores(va, predict(model, va))
        auc = float(roc_auc_score(v.label, v.p))
        thr = best_threshold(v)
        print(f"ep{ep} val video AUC {auc:.4f} thr {thr:.2f}", flush=True)
        log.append({"epoch": ep, "loss": tot / args.steps, "val_video_auc": auc, "threshold": thr})
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "threshold": thr}, out / "model.pt")

    ck = torch.load(out / "model.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    res = report(te, predict(model, te), ck["threshold"])
    print("TEST", json.dumps(res, indent=1), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    name = "swap_detector.json" if args.task == "swap" else f"{args.task}_detector.json"
    json.dump({"log": log, "test": res}, open(ROOT / "results" / name, "w"), indent=1)


if __name__ == "__main__":
    main()
