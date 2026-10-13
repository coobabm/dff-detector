"""Foundation-model face detector, following UnivFD (CVPR 2023) and B-Free (CVPR 2025).

  uv run fm_detector.py probe     # frozen CLIP ViT-L/14 and DINOv2-B features + linear layer (UnivFD)
  uv run fm_detector.py finetune  # DINOv2-B (registers), last blocks + head trained (B-Free style)

Why: an end-to-end CNN memorizes the artifacts of the generators it was trained on. A frozen
foundation-model feature space was not trained for this task, so a linear boundary in it transfers
better to unseen generators (UnivFD). B-Free adds content-matched fakes (gen_aligned.py: our own real
faces through SD/SDXL/FLUX autoencoders and SD2 inpainting) so content can not be a shortcut.

Data = everything collected so far, balanced per source: real photos, FairFace, Reddit photography,
OpenFake/LAION/Pexels reals, real video frames; fakes from SD/OpenAI/StyleGAN/AI video/face swap,
SFHQ (FLUX, SDXL, DALL-E 3), OpenFake (gpt-image, nano-banana, Imagen, MJ, Seedream ...), Reddit AI art,
aligned fakes. All are 256 px loose face crops, JPEG q95. Held-out sets include Rapidata (newest
generators, never trained on). Baseline: runs/all_shot. Saves runs/fm_*/ and results/fm_*.json.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import screenshot_finetune as S
from big_image import test_sets
from openai_curve import load_sets
from train import DATA, Faces, build_model, degrade
from train import eval_tf as b0_eval_tf

ROOT = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BACKBONES = {  # name -> (timm id, mean, std)
    "clip_l14": ("vit_large_patch14_clip_224.openai", (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
    "dinov2_b": ("vit_base_patch14_reg4_dinov2.lvd142m", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
EXTRA = ["reddit_manifest.csv", "hf_manifest.csv", "aligned_manifest.csv", "myphotos_manifest.csv"]
LORA = int(os.environ.get("FM_LORA", 0))  # LoRA rank on all blocks; 0 = unfreeze the last blocks instead
REAL_WEIGHT = float(os.environ.get("FM_REAL_WEIGHT", 1))
TAG = os.environ.get("FM_TAG", "")  # e.g. "_v2": new run folder, earlier models stay untouched
SWAPISH = ("ffpp", "ffusion", "hq_", "df40_inswap", "df40_simswap", "df40_uniface")


def tfs(name, train):
    _, mean, std = BACKBONES[name]
    if train:
        return T.Compose([T.Lambda(degrade), T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])


def backbone(name, num_classes=0):
    return timm.create_model(BACKBONES[name][0], pretrained=True, num_classes=num_classes, img_size=224)


def data():
    sets = load_sets()
    gan = pd.read_csv(DATA / "gan_manifest.csv")
    vid = S.load_video()
    extra = pd.concat([pd.read_csv(DATA / f) for f in EXTRA if (DATA / f).exists()], ignore_index=True)
    test_paths = set(sets["real_test"].path)
    ffhq = sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / "ffhq").glob("*.jpg"))
    pools = {
        "real": [("photos", sets["real_train"].assign(source="photos")),
                 ("ffhq_gan_reals", pd.concat([pd.DataFrame({"path": [p for p in ffhq if p not in test_paths], "label": 0, "source": "ffhq"}),
                                               gan[(gan.split == "train") & (gan.label == 0)]])),
                 ("video_real", vid[(vid.split == "train") & (vid.label == 0)])],
        # Face swaps get their own family: inside "all video fakes" they were only ~4% of fake samples.
        "fake": [("video_ai", vid[(vid.split == "train") & (vid.label == 1) & ~vid.group.str.startswith(SWAPISH)]),
                 ("video_swap", vid[(vid.split == "train") & (vid.label == 1) & vid.group.str.startswith(SWAPISH)]),
                 ("dff", sets["dff_fake_train"].assign(source="dff")),
                 ("openai", sets["oa_pool"].assign(source="openai")),
                 ("stylegan", gan[(gan.split == "train") & (gan.label == 1)])],
    }
    tr = extra[extra.split == "train"]
    # Families, so 60 small OpenFake generators do not each weigh as much as all video frames.
    # Inside a family, mixture() still gives each source an equal share.
    families = {
        "real": {"reddit_real": r"^rd_real_", "fairface": r"^hf_fairface", "openfake_real": r"^hf_of_real_",
                 "aligned_real": r"^aligned_real_resampled"},
        "fake": {"reddit_ai": r"^rd_(?!real_)", "sfhq": r"^hf_sfhq_", "openfake": r"^hf_of_(?!real_)",
                 "aligned_vae": r"^aligned_vae_", "aligned_inpaint": r"^aligned_inpaint"},
    }
    # The owner's camera originals (private, ~150 crops) share the Reddit-photography slot in training,
    # so they are not oversampled; they get their own private test set below (never in public tables).
    train_patterns = {**families["real"], "reddit_real": r"^rd_real_|^my_photos$"}
    for cls, fams in families.items():
        for fam, pattern in fams.items():
            pattern = train_patterns.get(fam, pattern) if cls == "real" else pattern
            g = tr[tr.source.str.contains(pattern, regex=True)]
            if len(g):
                pools[cls].append((fam, g))
    tests = test_sets(sets, gan[gan.split == "test"], vid[vid.split == "test"], extra.iloc[:0])
    te = extra[extra.split == "test"]
    for name, pattern in {**{f"{k}": v for k, v in families["real"].items()}, **families["fake"],
                          "rapidata_unseen": r"^hf_rapidata_", "my_photos": r"^my_photos$"}.items():
        g = te[te.source.str.contains(pattern, regex=True)]
        if len(g):
            tests[name] = g
    # Cap big sets (keeps the evaluation under ~15 min); photo_real stays whole for the threshold halves.
    tests = {k: (d if k == "photo_real" or len(d) <= 1500 else d.sample(1500, random_state=0)) for k, d in tests.items()}
    return pools, tests


def mixture(pools, n, seed):
    """n real + n fake rows: equal share per pool, and inside each pool equal share per source/group."""
    parts = []
    for cls in ("real", "fake"):
        k = n // len(pools[cls])
        for _, df in pools[cls]:
            by = "group" if "group" in df.columns and df.group.notna().all() else "source"
            parts.append(S.share(df, k, seed, by=by)[["path", "label"]])
    return pd.concat(parts, ignore_index=True)


@torch.no_grad()
def scores(model, df, tf, bs=64, feats=False):
    model.eval()
    out = []
    workers = int(os.environ.get("FM_WORKERS", 8))  # 0 when memory is tight: dead workers hang the loader
    for x, _ in DataLoader(Faces(df, tf), batch_size=bs, num_workers=workers):
        y = model(x.to(DEVICE))
        out.append((y if feats else torch.sigmoid(y.squeeze(1))).float().cpu())
    return torch.cat(out).numpy()


def summarize(name, sc, tests, half_a, half_b):
    thr = float(np.quantile(sc["photo_real"][half_a], .97))
    row = {"threshold": thr}
    for k, s in sc.items():
        s = s[half_b] if k == "photo_real" else s
        real = int(tests[k].label.iloc[0]) == 0
        row[k] = round(float(np.mean(s < thr) if real else np.mean(s > thr)), 3)
    print(name, json.dumps(row), flush=True)
    return row


def baseline(tests, halves):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(ROOT / "runs" / "all_shot" / "model.pt", map_location="cpu"))
    m.to(DEVICE)
    return summarize("all_shot", {k: scores(m, d, b0_eval_tf) for k, d in tests.items()}, tests, *halves)


def halves_of(tests):
    idx = np.random.default_rng(0).permutation(len(tests["photo_real"]))
    return idx[: len(idx) // 2], idx[len(idx) // 2:]


def probe(n=15000):
    from sklearn.linear_model import LogisticRegression

    pools, tests = data()
    halves = halves_of(tests)
    res = {"all_shot": baseline(tests, halves)}
    train = mixture(pools, n, 0)
    val = mixture(pools, n // 5, 99)
    val = val[~val.path.isin(set(train.path))]  # rows the probe never sees, for choosing C
    for name in BACKBONES:
        m = backbone(name).to(DEVICE)
        X = scores(m, train, tfs(name, False), feats=True)
        V = scores(m, val, tfs(name, False), feats=True)
        feats = {k: scores(m, d, tfs(name, False), feats=True) for k, d in tests.items()}
        mu, sd = X.mean(0), X.std(0) + 1e-6
        from sklearn.metrics import log_loss

        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, train.label.values)
            vloss = log_loss(val.label.values, clf.predict_proba((V - mu) / sd)[:, 1])
            print(f"{name} C={C} val log-loss {vloss:.4f}", flush=True)
            if best is None or vloss < best[0]:
                best = (vloss, C, clf)
        clf = best[2]
        r = summarize(f"probe_{name}_C{best[1]}", {k: clf.predict_proba((f - mu) / sd)[:, 1] for k, f in feats.items()},
                      tests, *halves)
        best = (best[0], best[1], clf, r)
        res[f"probe_{name}"] = {"C": best[1], **best[3]}
        out = ROOT / "runs" / f"fm_probe_{name}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"coef": torch.tensor(best[2].coef_), "intercept": torch.tensor(best[2].intercept_),
                    "mu": torch.tensor(mu), "sd": torch.tensor(sd), "backbone": BACKBONES[name][0]}, out / "probe.pt")
        del m
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "fm_probe.json", "w"), indent=1)


def plain_state(m):
    """State dict of a plain timm model (LoRA merged into the weights), loadable by app.py without peft."""
    if not LORA:
        return m.state_dict()
    import copy

    merged = copy.deepcopy(m).merge_and_unload()
    return {k: v.detach().cpu() for k, v in merged.state_dict().items()}


def finetune(name="dinov2_b", epochs=4, n=15000, bs=32, unfreeze=4):
    os.environ["DFF_CINEMATIC"] = "1"
    torch.manual_seed(0)
    pools, tests = data()
    halves = halves_of(tests)
    m = backbone(name, num_classes=1).to(DEVICE)
    for p in m.parameters():
        p.requires_grad = False
    if LORA:
        # LoRA on every block (attention qkv/proj + MLP): all layers adapt a little, the pretrained
        # weights stay frozen (B-Free fine-tunes the whole network; LoRA is the memory-friendly version).
        from peft import LoraConfig, get_peft_model

        m = get_peft_model(m, LoraConfig(r=LORA, lora_alpha=2 * LORA, lora_dropout=0.05,
                                         target_modules=["qkv", "proj", "fc1", "fc2"], modules_to_save=["head"]))
        for p in [*m.base_model.model.norm.parameters(), *(getattr(m.base_model.model, "fc_norm", nn.Identity()).parameters())]:
            p.requires_grad = True
        head = [p for n_, p in m.named_parameters() if p.requires_grad and "head" in n_]
        body_lr = 2e-4
    else:
        train_parts = [*m.blocks[-unfreeze:], m.norm, m.head] + ([m.fc_norm] if hasattr(m, "fc_norm") else [])
        for part in train_parts:
            for p in part.parameters():
                p.requires_grad = True
        head = [p for p in m.head.parameters()]
        body_lr = 1e-5 if name.startswith("clip") else 2e-5  # CLIP features are already good: move them less
    body = [p for p in m.parameters() if p.requires_grad and all(p is not h for h in head)]
    print(f"trainable {sum(p.numel() for p in body + head) / 1e6:.1f}M params (LoRA r={LORA}, real weight {REAL_WEIGHT})", flush=True)
    opt = torch.optim.AdamW([{"params": body, "lr": body_lr}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    steps = epochs * (2 * n // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    # Real images weigh REAL_WEIGHT times more than fakes (pos_weight scales the fake term), so a false
    # alarm on a real face costs more and the 97%-real operating point moves towards 0.5.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / REAL_WEIGHT, device=DEVICE))
    for ep in range(epochs):
        dl = DataLoader(Faces(mixture(pools, n, ep), tfs(name, True)), batch_size=bs, shuffle=True,
                        num_workers=8, drop_last=True)
        m.train()
        tot = 0.0
        for i, (x, y) in enumerate(dl):
            loss = loss_fn(m(x.to(DEVICE)).squeeze(1), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"ep{ep} {i}/{len(dl)} loss {tot / (i + 1):.4f}", flush=True)
        out = ROOT / "runs" / f"fm_ft_{name}{TAG}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(plain_state(m), out / "model.pt")
        (out / "arch.json").write_text(json.dumps({"timm": BACKBONES[name][0], "size": 224, "mean": BACKBONES[name][1],
                                                   "std": BACKBONES[name][2]}))
    r = summarize(f"ft_{name}", {k: scores(m, d, tfs(name, False)) for k, d in tests.items()}, tests, *halves)
    out = ROOT / "runs" / f"fm_ft_{name}{TAG}"
    (out / "threshold.json").write_text(json.dumps({"threshold": round(r["threshold"], 3),
                                                    "note": "97th percentile on half of real_test; see results/fm_finetune.json"}))
    json.dump({"all_shot": baseline(tests, halves), f"ft_{name}": r}, open(ROOT / "results" / f"fm_finetune_{name}{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "finetune":
        finetune(*(sys.argv[2:3] or ["dinov2_b"]), **({"epochs": int(sys.argv[3])} if len(sys.argv) > 3 else {}),
                 **({"unfreeze": int(sys.argv[4])} if len(sys.argv) > 4 else {}))
    else:
        probe()
