"""Bigger image detector: EfficientNet-B4 at 380 px on loose face crops, all sources together.

  uv run training/big_image.py [--epochs 5] [--per-class 12000]

Same data as screenshot_finetune.py (real photos, StyleGAN, DFF, OpenAI, AI-video / face-swap frames)
plus any extra manifests that exist (data/reddit_manifest.csv, data/hf_manifest.csv: path, source,
label, split, with split by post / generator item). Every source is stored as 256 px loose crops, so
real and fake go through the same 256 -> 380 resize: resolution can not become a shortcut.
Starts from ImageNet weights. Compares against runs/all_shot on every held-out set.
Saves runs/big_b4/{model.pt,threshold.json,arch.json} and results/big_image.json.
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import screenshot_finetune as S
from openai_curve import load_sets
from train import DATA, Faces, build_model, predict
from train import eval_tf as b0_eval_tf
from train_swap import build as build_b4
from train_swap import eval_tf as b4_eval_tf
from train_swap import train_tf as b4_train_tf

ROOT = Path(__file__).resolve().parents[1]
EXTRA = [DATA / "reddit_manifest.csv", DATA / "hf_manifest.csv", DATA / "hf_real_manifest.csv"]


@torch.no_grad()
def scores(model, df, tf, device, bs=64):
    model.eval()
    out = [torch.sigmoid(model(x.to(device)).squeeze(1)).cpu()
           for x, _ in DataLoader(Faces(df, tf), batch_size=bs, num_workers=8)]
    return torch.cat(out).numpy() if out else np.zeros(0)


def test_sets(sets, gan_te, vid_te, extra_te):
    swapish = ("ffpp", "ffusion", "hq_", "df40_inswap", "df40_simswap", "df40_uniface")
    t = {"photo_real": sets["real_test"], "video_real_frames": vid_te[vid_te.label == 0],
         "ai_video_frames": vid_te[(vid_te.label == 1) & ~vid_te.group.str.startswith(swapish)],
         "swap_frames": vid_te[(vid_te.label == 1) & vid_te.group.str.startswith(swapish)],
         "openai": sets["oa_test"], "dff_fake": sets["dff_fake_test"],
         "stylegan": gan_te[gan_te.label == 1], "gan_set_real": gan_te[gan_te.label == 0]}
    for src, g in extra_te.groupby("source"):
        t[src] = g
    return t


def report(name, sc, sets_, half_a, half_b):
    """Threshold = 97th percentile on half A of the held-out real photos; numbers on everything else."""
    thr = float(np.quantile(sc["photo_real"][half_a], .97))
    row = {"threshold": thr}
    for k, s in sc.items():
        s = s[half_b] if k == "photo_real" else s
        real = int(sets_[k].label.iloc[0]) == 0
        row[k] = {"n": len(s), "acc" if real else "recall": float(np.mean(s < thr) if real else np.mean(s > thr))}
    print(name, json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    os.environ["DFF_CINEMATIC"] = "1"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    sets = load_sets()
    gan = pd.read_csv(DATA / "gan_manifest.csv")
    gan_tr, gan_te = gan[gan.split == "train"], gan[gan.split == "test"]
    vid = S.load_video()
    vid_tr, vid_te = vid[vid.split == "train"], vid[vid.split == "test"]
    extra = pd.concat([pd.read_csv(f) for f in EXTRA if f.exists()] or [pd.DataFrame(columns=["path", "source", "label", "split"])])
    extra_tr, extra_te = extra[extra.split == "train"], extra[extra.split == "test"]
    print("extra sources", extra.groupby(["source", "label", "split"]).size().unstack(fill_value=0).to_string(), flush=True)

    tests = test_sets(sets, gan_te, vid_te, extra_te)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tests["photo_real"]))
    half_a, half_b = idx[: len(idx) // 2], idx[len(idx) // 2:]

    # Baseline: the deployed image model.
    shot = build_model(pretrained=False)
    shot.load_state_dict(torch.load(ROOT / "runs" / "all_shot" / "model.pt", map_location="cpu"))
    shot.to(device)
    base = report("all_shot", {k: scores(shot, d, b0_eval_tf, device) for k, d in tests.items()}, tests, half_a, half_b)
    del shot

    test_paths = set(sets["real_test"].path)
    ffhq = sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / "ffhq").glob("*.jpg"))
    ffhq_extra = pd.DataFrame({"path": [p for p in ffhq if p not in test_paths], "label": 0, "source": "ffhq"})
    v_real, v_fake = vid_tr[vid_tr.label == 0], vid_tr[vid_tr.label == 1]
    g_real, g_fake = gan_tr[gan_tr.label == 0], gan_tr[gan_tr.label == 1]
    x_real, x_fake = extra_tr[extra_tr.label == 0], extra_tr[extra_tr.label == 1]
    n = args.per_class

    model = build_b4(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (2 * n // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    loss_fn = nn.BCEWithLogitsLoss()
    cols = ["path", "label"]
    for ep in range(args.epochs):
        real_parts = [sets["real_train"].sample(n * 3 // 10, replace=True, random_state=ep),
                      ffhq_extra.sample(n // 10, replace=True, random_state=ep),
                      g_real.sample(n // 10, replace=True, random_state=ep),
                      S.share(v_real, n * 3 // 10, ep)]
        if len(x_real):
            real_parts.append(S.share(x_real, n * 2 // 10, ep, by="source"))
        else:
            real_parts.append(sets["real_train"].sample(n * 2 // 10, replace=True, random_state=ep + 100))
        fake_parts = [S.share(v_fake, n * 3 // 10, ep),
                      S.share(g_fake, n // 10, ep, by="source"),
                      sets["dff_fake_train"].sample(n // 10, random_state=ep),
                      sets["oa_pool"].sample(n // 10, replace=True, random_state=ep)]
        if len(x_fake):
            fake_parts.append(S.share(x_fake, n * 4 // 10, ep, by="source"))
        else:
            fake_parts.append(S.share(v_fake, n * 4 // 10, ep + 100))
        df = pd.concat([*(p[cols] for p in real_parts), *(p[cols] for p in fake_parts)])
        dl = DataLoader(Faces(df, b4_train_tf), batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True)
        model.train()
        tot = 0.0
        for i, (x, y) in enumerate(dl):
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"ep{ep} {i}/{len(dl)} loss {tot / (i + 1):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        out = ROOT / "runs" / "big_b4"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")

    big = report("big_b4", {k: scores(model, d, b4_eval_tf, device) for k, d in tests.items()}, tests, half_a, half_b)
    out = ROOT / "runs" / "big_b4"
    (out / "arch.json").write_text(json.dumps({"arch": "efficientnet_b4.ra2_in1k", "size": 380}))
    (out / "threshold.json").write_text(json.dumps({"threshold": round(big["threshold"], 3),
                                                    "note": "97th percentile on half of real_test; see results/big_image.json"}))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"all_shot": base, "big_b4": big, "args": vars(args)}, open(ROOT / "results" / "big_image.json", "w"), indent=1)


if __name__ == "__main__":
    main()
