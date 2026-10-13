"""Teach the image model screenshots of AI videos without breaking real photos.

  uv run screenshot_finetune.py

Starts from runs/all_image (AI images + StyleGAN). Each epoch is balanced real/fake:
  real: the real photo sources + extra FFHQ + StyleGAN-set reals + real video frames
  fake: AI-video / face-swap / talking-head frames + StyleGAN + DFF fakes + OpenAI images
Video frames are loose face crops (video_scan.faces_in), the same crop app.py uses for
uploaded images. Frames are split by video, never by frame. Tests every source on held-out
data before and after. Saves runs/all_shot/model.pt and results/screenshot_finetune.json.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openai_curve import load_sets
from train import DATA, Faces, build_model, predict, train_tf

ROOT = Path(__file__).parent
EPOCHS = 4
PER_CLASS = 10000


def load_video():
    parts = [pd.read_csv(f) for f in sorted((DATA / "video_manifest").glob("*.csv"))]
    vid = pd.concat(parts, ignore_index=True)
    # Rapidata prompt folders (rp_0001 ...) are tiny; pool them so they do not get a full share each.
    vid["group"] = vid.source.map(lambda s: "rp_prompts" if re.fullmatch(r"rp_\d+", s) else s)
    return vid


def share(df, n, seed, by="group"):
    """n rows with an equal share per group (sampled with replacement)."""
    groups = list(df.groupby(by))
    return pd.concat([g.sample(max(1, n // len(groups)), replace=True, random_state=seed) for _, g in groups])


def evaluate(model, sets, gan_te, vid_te, device):
    res = {}
    p = predict(model, vid_te, device)
    v = vid_te.assign(p=p)
    res["video_frames"] = {s: {"n": len(g), ("acc" if g.label.iloc[0] == 0 else "recall"):
                               float(((g.p > .5) == (g.label == 1)).mean())} for s, g in v.groupby("group")}
    real_v, fake_v = v[v.label == 0], v[v.label == 1]
    res["video_real_frame_acc"] = float((real_v.p < .5).mean())
    res["video_fake_frame_recall"] = float((fake_v.p > .5).mean())
    for src, g in gan_te.groupby("source"):
        q = predict(model, g, device)
        res["gan_" + src] = float(((q > .5) == (g.label.values == 1)).mean())
    po, pr, pdf = (predict(model, sets[k], device) for k in ("oa_test", "real_test", "dff_fake_test"))
    rt = sets["real_test"].assign(p=pr)
    res["openai_recall"] = float((po > .5).mean())
    res["photo_real_acc"] = float((pr < .5).mean())
    res["photo_real_acc_by_source"] = {k: float((g.p < .5).mean()) for k, g in rt.groupby("source")}
    res["dff_fake_recall"] = float((pdf > .5).mean())
    return res


def main():
    os.environ["DFF_CINEMATIC"] = "1"  # film-look augmentation, inherited by DataLoader workers
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    sets = load_sets()
    gan = pd.read_csv(DATA / "gan_manifest.csv")
    gan_tr, gan_te = gan[gan.split == "train"], gan[gan.split == "test"]
    vid = load_video()
    vid_tr, vid_te = vid[vid.split == "train"], vid[vid.split == "test"]
    print(vid.groupby(["group", "split"]).size().unstack(fill_value=0).to_string(), flush=True)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(ROOT / "runs" / "all_image" / "model.pt", map_location="cpu"))
    model.to(device)
    before = evaluate(model, sets, gan_te, vid_te, device)
    print("BEFORE", json.dumps(before), flush=True)

    test_paths = set(sets["real_test"].path)
    ffhq = sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / "ffhq").glob("*.jpg"))
    ffhq_extra = pd.DataFrame({"path": [p for p in ffhq if p not in test_paths], "label": 0, "source": "ffhq"})
    v_real, v_fake = vid_tr[vid_tr.label == 0], vid_tr[vid_tr.label == 1]
    g_real, g_fake = gan_tr[gan_tr.label == 0], gan_tr[gan_tr.label == 1]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    cols = ["path", "label"]
    for ep in range(EPOCHS):
        real = pd.concat([sets["real_train"].sample(PER_CLASS * 3 // 8, replace=True, random_state=ep),
                          ffhq_extra.sample(PER_CLASS // 8, replace=True, random_state=ep),
                          g_real.sample(PER_CLASS // 8, replace=True, random_state=ep),
                          share(v_real, PER_CLASS * 3 // 8, ep)])
        fake = pd.concat([share(v_fake, PER_CLASS // 2, ep),
                          share(g_fake, PER_CLASS // 6, ep, by="source"),
                          sets["dff_fake_train"].sample(PER_CLASS // 6, random_state=ep),
                          sets["oa_pool"].sample(PER_CLASS // 6, replace=True, random_state=ep)])
        dl = DataLoader(Faces(pd.concat([real[cols], fake[cols]]), train_tf), batch_size=64, shuffle=True,
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

    after = evaluate(model, sets, gan_te, vid_te, device)
    print("AFTER", json.dumps(after), flush=True)
    out = ROOT / "runs" / "all_shot"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"before": before, "after": after}, open(ROOT / "results" / "screenshot_finetune.json", "w"), indent=1)


if __name__ == "__main__":
    main()
