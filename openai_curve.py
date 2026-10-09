"""Learning curve: how many OpenAI images does the detector need?

Fine-tunes the DFF `all` model with N OpenAI fakes (N = 0, 50, 100, 200, 500, all),
always tests on the same held-out 200 OpenAI images.
Reals: DFF wiki + FFHQ + Pexels + PPR10k + Flickr30k + EasyPortrait,
so "sharp, modern photo" alone does not mean fake.
DFF fakes stay in training so old generators are not forgotten.

  uv run openai_curve.py
"""
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from train import DATA, FAKES, Faces, build_model, predict, train_tf

ROOT = Path(__file__).parent
OUT = ROOT / "runs_openai"
REALS = ["ffhq", "pexels", "ppr10k", "flickr30k", "easyportrait"]
SIZES = [0, 50, 100, 200, 500, 10_000]
PER_CLASS = 6000  # images per class per epoch
EPOCHS = 3


def frame(paths, label, source):
    return pd.DataFrame({"path": paths, "label": label, "source": source})


def load_sets():
    rng = random.Random(0)
    meta = json.load(open(DATA / "openai" / "meta.json"))
    ids = sorted(meta)
    rng.shuffle(ids)
    oa_test, oa_pool = ids[:200], ids[200:]
    oa = lambda xs: frame([f"img256/openai/{i}.jpg" for i in xs], 1, [meta[i]["model"] for i in xs])

    m = pd.read_csv(DATA / "manifest.csv")
    tr, te = m[m.split == "train"], m[m.split == "test"]
    per_src = PER_CLASS // (len(REALS) + 1)
    real_train = [tr[tr.source == "wiki"].sample(per_src, random_state=0)]
    real_test = {"wiki": te[te.source == "wiki"].sample(500, random_state=0)}
    for src in REALS:
        files = sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / src).glob("*.jpg"))
        rng.shuffle(files)
        real_test[src] = frame(files[:500], 0, src)
        real_train.append(frame(files[500:500 + per_src], 0, src))
    sets = {
        "oa_pool": oa(oa_pool),
        "oa_test": oa(oa_test),
        "real_train": pd.concat(real_train),
        "real_test": pd.concat(real_test.values()),
        "dff_fake_train": pd.concat([tr[tr.source == s].sample(PER_CLASS // 3, random_state=0) for s in FAKES]),
        "dff_fake_test": pd.concat([te[te.source == s].sample(500, random_state=0) for s in FAKES]),
    }
    return sets


def run(n, sets, device):
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "runs" / "all" / "model.pt", map_location="cpu"))
    model.to(device)
    oa_train = sets["oa_pool"].sample(min(n, len(sets["oa_pool"])), random_state=0) if n else sets["oa_pool"][:0]
    n = len(oa_train)
    if n:
        # Half the fake slots go to OpenAI images (oversampled), half to DFF fakes.
        fakes = pd.concat([oa_train.sample(PER_CLASS // 2, replace=True, random_state=0),
                           sets["dff_fake_train"].sample(PER_CLASS // 2, random_state=0)])
        train = pd.concat([sets["real_train"], fakes])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(EPOCHS):
            model.train()
            dl = DataLoader(Faces(train, train_tf), batch_size=64, shuffle=True, num_workers=8, drop_last=True)
            for x, y in dl:
                loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f"  n={n} ep{ep} loss {loss.item():.4f}", flush=True)

    p = {k: predict(model, sets[k], device) for k in ("oa_test", "real_test", "dff_fake_test")}
    reals = list(p["real_test"])
    oa_test = sets["oa_test"].assign(p=p["oa_test"])
    real_test = sets["real_test"].assign(p=p["real_test"])
    res = {
        "n_train": n,
        "openai_auc": float(roc_auc_score([0] * len(reals) + [1] * len(p["oa_test"]), reals + list(p["oa_test"]))),
        "openai_recall": float((p["oa_test"] > 0.5).mean()),
        "openai_recall_by_model": {k: float((g.p > 0.5).mean()) for k, g in oa_test.groupby("source")},
        "real_acc": float((p["real_test"] < 0.5).mean()),
        "real_acc_by_source": {k: float((g.p < 0.5).mean()) for k, g in real_test.groupby("source")},
        "dff_fake_recall": float((p["dff_fake_test"] > 0.5).mean()),
    }
    torch.save(model.state_dict(), OUT / f"n{n}.pt")
    return res


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    sets = load_sets()
    print({k: len(v) for k, v in sets.items()})
    results = []
    for n in SIZES:
        if n > len(sets["oa_pool"]) and n != SIZES[-1]:
            continue
        r = run(n, sets, device)
        print(json.dumps(r), flush=True)
        results.append(r)
        json.dump(results, open(OUT / "curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
