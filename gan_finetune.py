"""Teach the image model StyleGAN faces without forgetting the rest.

  uv run gan_finetune.py

Starts from runs/all_openai. Each epoch (balanced real/fake):
  real: the 6 real photo sources + all non-test FFHQ photos + FFHQ reals from the 140k set
  fake: StyleGAN (tpdne, sg2face, sg3, k140_fake; equal share) + DFF fakes + OpenAI images
Tests on held-out images of every source. Saves runs/all_image/model.pt and
results/gan_finetune.json (before/after).
"""
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openai_curve import load_sets
from train import DATA, Faces, build_model, predict, train_tf

ROOT = Path(__file__).parent
EPOCHS = 4
PER_CLASS = 8000


def evaluate(model, sets, gan_te, device):
    res = {}
    for src, g in gan_te.groupby("source"):
        p = predict(model, g, device)
        key = "acc" if g.label.iloc[0] == 0 else "recall"
        res[src] = {key: float(((p > 0.5) == (g.label.values == 1)).mean()), "n": len(g)}
    po = predict(model, sets["oa_test"], device)
    pr = predict(model, sets["real_test"], device)
    pdf = predict(model, sets["dff_fake_test"], device)
    rt = sets["real_test"].assign(p=pr)
    res["openai_recall"] = float((po > 0.5).mean())
    res["photo_real_acc"] = float((pr < 0.5).mean())
    res["photo_real_acc_by_source"] = {k: float((g.p < 0.5).mean()) for k, g in rt.groupby("source")}
    res["dff_fake_recall"] = float((pdf > 0.5).mean())
    return res


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    sets = load_sets()
    gan = pd.read_csv(DATA / "gan_manifest.csv")
    gan_tr, gan_te = gan[gan.split == "train"], gan[gan.split == "test"]
    print(gan.groupby(["source", "split"]).size().unstack(fill_value=0).to_string(), flush=True)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "runs" / "all_openai" / "model.pt", map_location="cpu"))
    model.to(device)
    before = evaluate(model, sets, gan_te, device)
    print("BEFORE", json.dumps(before), flush=True)

    g_fake = gan_tr[gan_tr.label == 1]
    g_real = gan_tr[gan_tr.label == 0]
    # StyleGAN was trained on FFHQ, so real FFHQ faces are the hardest negatives:
    # add every FFHQ photo not used for testing.
    test_paths = set(sets["real_test"].path)
    ffhq = sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / "ffhq").glob("*.jpg"))
    ffhq_extra = pd.DataFrame({"path": [p for p in ffhq if p not in test_paths], "label": 0, "source": "ffhq"})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        real = pd.concat([sets["real_train"].sample(PER_CLASS // 2, replace=True, random_state=ep),
                          ffhq_extra.sample(PER_CLASS // 4, replace=True, random_state=ep),
                          g_real.sample(PER_CLASS // 4, replace=True, random_state=ep)])
        fake = pd.concat([*(g.sample(PER_CLASS // 2 // g_fake.source.nunique(), replace=True, random_state=ep)
                            for _, g in g_fake.groupby("source")),
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

    after = evaluate(model, sets, gan_te, device)
    print("AFTER", json.dumps(after), flush=True)
    out = ROOT / "runs" / "all_image"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    json.dump({"before": before, "after": after}, open(ROOT / "results" / "gan_finetune.json", "w"), indent=1)


if __name__ == "__main__":
    main()
