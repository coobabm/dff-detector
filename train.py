"""Train a real-vs-fake face classifier on DeepFakeFace.

Example:
  uv run train.py --fakes text2img --name t2i       # train on A only
  uv run train.py --fakes text2img inpainting insight --name all

After training, the model is tested on every fake source (cross-generator test).
"""
import argparse
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FAKES = ["text2img", "inpainting", "insight"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def cinematic(im: Image.Image) -> Image.Image:
    """Film look: darker gamma, lower saturation, teal/orange grade, grain."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a ** random.uniform(1.3, 2.4)) * random.uniform(0.35, 0.8)
    gray = a.mean(axis=2, keepdims=True)
    a = gray + (a - gray) * random.uniform(0.5, 1.0)
    lum = gray
    teal, orange = np.array([0.0, 0.08, 0.12]), np.array([0.12, 0.06, 0.0])
    s = random.uniform(0.0, 1.0)
    a = a + s * ((1 - lum) * teal + lum * orange)
    a = a + np.random.normal(0, random.uniform(0.0, 0.03), a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def degrade(im: Image.Image) -> Image.Image:
    """Random real-world degradations, applied to both classes equally."""
    # Opt-in (env DFF_CINEMATIC=1) so earlier DFF/OpenAI experiments stay reproducible.
    if os.environ.get("DFF_CINEMATIC") == "1" and random.random() < 0.3:
        im = cinematic(im)
    if random.random() < 0.2:
        s = random.randint(96, 200)
        im = im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BILINEAR)
    if random.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    if random.random() < 0.3:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(50, 95))
        im = Image.open(buf).convert("RGB")
    return im


train_tf = T.Compose([
    T.Lambda(degrade),
    T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class Faces(Dataset):
    def __init__(self, df, tf):
        self.paths = df.path.tolist()
        self.labels = df.label.tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(DATA / self.paths[i]).convert("RGB")
        return self.tf(im), torch.tensor(self.labels[i], dtype=torch.float32)


def build_model(pretrained=True):
    return timm.create_model("efficientnet_b0.ra_in1k", pretrained=pretrained, num_classes=1)


@torch.no_grad()
def predict(model, df, device, bs=128):
    model.eval()
    dl = DataLoader(Faces(df, eval_tf), batch_size=bs, num_workers=8)
    out = []
    for x, _ in dl:
        out.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
    return torch.cat(out).numpy()


def evaluate(model, df, device):
    """AUC/acc of real vs each fake source, on given split."""
    real = df[df.source == "wiki"]
    p_real = predict(model, real, device)
    res = {"real_acc": float((p_real < 0.5).mean())}
    for src in FAKES:
        fake = df[df.source == src]
        if fake.empty:
            continue
        p_fake = predict(model, fake, device)
        y = [0] * len(p_real) + [1] * len(p_fake)
        p = list(p_real) + list(p_fake)
        res[src] = {
            "auc": float(roc_auc_score(y, p)),
            "acc": float(accuracy_score(y, [v > 0.5 for v in p])),
            "fake_recall": float((p_fake > 0.5).mean()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", nargs="+", default=["text2img"], choices=FAKES)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train images per class")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    m = pd.read_csv(DATA / "manifest.csv")
    tr, va, te = (m[m.split == s] for s in ("train", "val", "test"))
    tr_real = tr[tr.source == "wiki"]
    tr_fake = tr[tr.source.isin(args.fakes)]
    if args.limit:
        tr_real, tr_fake = tr_real.sample(args.limit, random_state=0), tr_fake.sample(args.limit, random_state=0)
        va = va.groupby("source").sample(args.limit // 4, random_state=0)
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = 2 * len(tr_real) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    va_trained = va[va.source.isin(["wiki", *args.fakes])]

    best, log = -1, []
    for ep in range(args.epochs):
        # Balanced epoch: all real + same number of fakes resampled each epoch.
        fake_ep = tr_fake.sample(len(tr_real), replace=len(tr_fake) < len(tr_real), random_state=ep)
        dl = DataLoader(Faces(pd.concat([tr_real, fake_ep]), train_tf), batch_size=args.bs, shuffle=True,
                        num_workers=8, drop_last=True, persistent_workers=False)
        model.train()
        t0, tot = time.time(), 0.0
        for i, (x, y) in enumerate(dl):
            if i >= steps_per_epoch:
                break
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if i % 50 == 0:
                print(f"ep{ep} {i}/{steps_per_epoch} loss {tot / (i + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        v = evaluate(model, va_trained, device)
        score = sum(v[s]["auc"] for s in args.fakes) / len(args.fakes)
        print(f"ep{ep} val {json.dumps(v)}", flush=True)
        log.append({"epoch": ep, "train_loss": tot / steps_per_epoch, "val": v})
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    test = evaluate(model, te, device)
    print("TEST", json.dumps(test, indent=1))
    json.dump({"fakes": args.fakes, "log": log, "test": test}, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
