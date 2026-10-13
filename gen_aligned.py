"""Content-matched fakes from our own real faces (the B-Free idea, CVPR 2025).

  uv run gen_aligned.py [--train 3000 --test 600]

Each sampled real crop is turned into a fake that keeps its exact content, so a detector can not
use content (who, where, which dataset) as a shortcut and has to learn generator traces:
  vae_sd15  - SD 1.x/2.x autoencoder round trip       (stabilityai/sd-vae-ft-mse)
  vae_sdxl  - SDXL autoencoder round trip             (madebyollin/sdxl-vae-fp16-fix)
  vae_flux  - FLUX 16-channel autoencoder round trip  (ostris/Flex.1-alpha vae = FLUX.1 ae, ungated)
  inpaint   - SD2 inpainting of a face/random region with an empty prompt (sd2-community mirror), at 512 px
Inpainting runs at 512, so its real partner is also resized 256->512->256 (real_resampled) to cancel
the resize trace. Splits follow the real image: fakes of test reals are test.
Writes data/img256/aligned/<method>/... and data/aligned_manifest.csv (path, source, label, split, real).
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter

import screenshot_finetune as S
from openai_curve import load_sets
from train import DATA

OUT = DATA / "img256" / "aligned"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VAES = {"vae_sd15": ("stabilityai/sd-vae-ft-mse", None), "vae_sdxl": ("madebyollin/sdxl-vae-fp16-fix", None),
        "vae_flux": ("ostris/Flex.1-alpha", "vae"),
        # round 2: newer autoencoders (FLUX.2 32-channel, Qwen-Image video-style VAE)
        "vae_flux2": ("black-forest-labs/FLUX.2-klein-4B", "vae", "AutoencoderKLFlux2"),
        "vae_qwen": ("Qwen/Qwen-Image", "vae", "AutoencoderKLQwenImage")}


def reals(n_train, n_test, seed=0):
    sets = load_sets()
    vid = S.load_video()
    vr = vid[vid.label == 0]
    parts = []
    for split, n, photos in (("train", n_train, sets["real_train"]), ("test", n_test, sets["real_test"])):
        v = vr[vr.split == split]
        parts.append(pd.concat([photos.sample(n * 2 // 3, random_state=seed)[["path"]],
                                v.sample(n // 3, random_state=seed)[["path"]]]).assign(split=split))
    df = pd.concat(parts, ignore_index=True)
    df["method"] = [list(VAES)[i % 3] if i % 4 else "inpaint" for i in range(len(df))]
    return df


def reals2(n_train, n_test, seed=1):
    """Round 2: more varied reals (photos, video frames, FairFace, Reddit photography), new seed."""
    sets = load_sets()
    vid = S.load_video()
    hf = pd.read_csv(DATA / "hf_manifest.csv")
    rd = pd.read_csv(DATA / "reddit_manifest.csv")
    pools = {"train": [sets["real_train"], vid[(vid.label == 0) & (vid.split == "train")],
                       hf[(hf.source == "hf_fairface") & (hf.split == "train")], rd[(rd.label == 0) & (rd.split == "train")]],
             "test": [sets["real_test"], vid[(vid.label == 0) & (vid.split == "test")],
                      hf[(hf.source == "hf_fairface") & (hf.split == "test")], rd[(rd.label == 0) & (rd.split == "test")]]}
    parts = [pd.concat([d.sample(min(len(d), n // 4), random_state=seed)[["path"]] for d in pools[split]]).assign(split=split)
             for split, n in (("train", n_train), ("test", n_test))]
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    methods = ["vae_flux2", "vae_qwen", "inpaint", "vae_sd15", "vae_sdxl", "vae_flux", "inpaint"]
    df["method"] = [methods[i % len(methods)] for i in range(len(df))]
    return df


def out_path(method, real):
    return OUT / method / (real.replace("/", "__").rsplit(".", 1)[0] + ".jpg")


@torch.no_grad()
def run_vae(df, method):
    import diffusers

    repo, sub, *cls = VAES[method]
    vae = getattr(diffusers, cls[0] if cls else "AutoencoderKL").from_pretrained(repo, subfolder=sub).to(DEVICE).eval()
    video = method == "vae_qwen"  # Qwen-Image VAE takes (B, C, T, H, W)
    todo = [p for p in df.path if not out_path(method, p).exists()]
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(DATA / p).convert("RGB").resize((256, 256)),
                                                     dtype=np.float32) / 127.5 - 1).permute(2, 0, 1) for p in batch])
        x = x.to(DEVICE)
        x = x.unsqueeze(2) if video else x
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
        y = y[:, :, 0] if video else y
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for p, a in zip(batch, y):
            dst = out_path(method, p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a).save(dst, quality=95)
        if i % 320 == 0:
            print(method, i, "/", len(todo), flush=True)
    del vae


def mask_for(rng):
    """Face-centred ellipse (most inpainting fakes edit the face) or a random blob."""
    m = Image.new("L", (512, 512), 0)
    d = ImageDraw.Draw(m)
    if rng.random() < 0.7:
        cx, cy, rx, ry = 256 + rng.randint(-30, 30), 230 + rng.randint(-30, 30), rng.randint(70, 130), rng.randint(90, 160)
    else:
        cx, cy, rx, ry = rng.randint(100, 412), rng.randint(100, 412), rng.randint(50, 150), rng.randint(50, 150)
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return m.filter(ImageFilter.GaussianBlur(8))


def run_inpaint(df, steps=15):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained("sd2-community/stable-diffusion-2-inpainting",
                                                          safety_checker=None).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    rng = random.Random(0)
    todo = [p for p in df.path if not out_path("inpaint", p).exists()]
    for i, p in enumerate(todo):
        src = Image.open(DATA / p).convert("RGB").resize((256, 256))
        big = src.resize((512, 512), Image.BICUBIC)
        g = torch.Generator("cpu").manual_seed(i)
        fake = pipe(prompt="", image=big, mask_image=mask_for(rng), num_inference_steps=steps,
                    guidance_scale=1.0, generator=g).images[0]
        for method, im in (("inpaint", fake), ("real_resampled", big)):
            dst = out_path(method, p)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.resize((256, 256), Image.BICUBIC).save(dst, quality=95)
        if i % 25 == 0:
            print("inpaint", i, "/", len(todo), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-inpaint", type=int, default=300)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    df = reals(args.train, args.test) if args.round == 1 else reals2(args.train, args.test)
    # SD2 inpainting is slow on a laptop GPU (~30 s/image with other jobs running): cap it, keep the split ratio.
    inp = df[df.method == "inpaint"]
    keep = pd.concat([inp[inp.split == "train"].head(args.max_inpaint * 4 // 5), inp[inp.split == "test"].head(args.max_inpaint // 5)])
    df = df[(df.method != "inpaint") | df.index.isin(keep.index)]
    for method in args.only or [*VAES, "inpaint"]:
        sub = df[df.method == method]
        run_inpaint(sub) if method == "inpaint" else run_vae(sub, method)
    rows = []
    for r in df.itertuples():
        rows.append({"path": str(out_path(r.method, r.path).relative_to(DATA)), "source": f"aligned_{r.method}",
                     "label": 1, "split": r.split, "real": r.path})
        if r.method == "inpaint":
            rows.append({"path": str(out_path("real_resampled", r.path).relative_to(DATA)), "source": "aligned_real_resampled",
                         "label": 0, "split": r.split, "real": r.path})
    man = pd.DataFrame(rows)
    old = DATA / "aligned_manifest.csv"
    if old.exists():  # rounds accumulate
        man = pd.concat([pd.read_csv(old), man], ignore_index=True).drop_duplicates("path")
    man = man[[(DATA / p).exists() for p in man.path]]
    man.to_csv(old, index=False)
    print(man.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
