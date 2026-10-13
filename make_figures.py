"""README figures for the latest round (CLIP LoRA + DINOv2 ensemble).

  uv run make_figures.py   # -> docs/round_*.png / .jpg

Needs the cached held-out scores of the ensemble members (fm_compare.py) and logs/fm_ft_clip_lora.txt.
Sample tiles are drawn at random (fixed seed) from the held-out test sets, not picked by hand; the
number under each tile is the ensemble score, red when above the threshold. Only sets whose licence
allows showing images are used (Rapidata, SFHQ, OpenAI images made by this project, OpenFake, Pexels,
FairFace).
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import fm_detector as F
from fm_compare import model_scores

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"
RED, GREEN, INK, GREY = "#C8372D", "#1E7F4F", "#1D1F21", "#9AA0A6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def ensemble_scores(tests):
    e = json.loads((ROOT / "runs" / "fm_ens" / "ensemble.json").read_text())
    cache = [model_scores(m, tests) for m in e["members"]]
    return {k: np.mean([c[k] for c in cache], 0) for k in tests}, e["threshold"]


def pick(df, sc, source, rng):
    idx = np.flatnonzero(df.source.values == source)
    i = rng.choice(idx)
    return df.path.values[i], sc[i]


def samples(tests, ens, thr):
    rng = np.random.default_rng(3)
    rows = [
        ("Never trained on (Rapidata)", "rapidata_unseen", [
            ("hf_rapidata_ideogram_ai_ideogram_v3_balanced", "Ideogram 3"), ("hf_rapidata_luma_photon", "Luma Photon"),
            ("hf_rapidata_bytedance_seedream_3", "Seedream 3"), ("hf_rapidata_google_imagen_4_fast", "Imagen 4 Fast"),
            ("hf_rapidata_recraft_ai_recraft_v3", "Recraft v3"), ("hf_rapidata_minimax_image_01", "MiniMax Image-01")]),
        ("Trained generators", None, [
            ("sfhq:hf_sfhq_flux1_pro", "FLUX.1 pro"), ("sfhq:hf_sfhq_sdxl", "SDXL"), ("sfhq:hf_sfhq_dalle3", "DALL-E 3"),
            ("openai:gpt-image-2", "gpt-image-2"), ("openfake:hf_of_midjourney_6", "Midjourney 6"),
            ("openfake:hf_of_imagen_4.0", "Imagen 4")]),
        ("Real photos", None, [
            ("photo_real:pexels", "Pexels"), ("photo_real:pexels", "Pexels"), ("openfake_real:hf_of_real_pexels", "Pexels"),
            ("fairface:hf_fairface", "FairFace"), ("fairface:hf_fairface", "FairFace"), ("fairface:hf_fairface", "FairFace")]),
    ]
    fig, axes = plt.subplots(3, 6, figsize=(12, 7.2))
    for r, (title, default_set, tiles) in enumerate(rows):
        for c, (key, label) in enumerate(tiles):
            s, src = (key.split(":") if ":" in key else (default_set, key))
            path, score = pick(tests[s], ens[s], src, rng)
            ax = axes[r, c]
            ax.imshow(Image.open(DATA / path).convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(label, fontsize=9.5, color=INK)
            ax.set_xlabel(f"{score:.2f}", color=RED if score > thr else GREEN, fontsize=12, fontweight="bold")
        axes[r, 0].text(-0.12, 0.5, title, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=11, fontweight="bold", color=INK)
    fig.suptitle(f"Random held-out samples, ensemble score (red = above threshold {thr})", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "round_samples.jpg", dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def aligned(tests):
    """Real crop, its content-matched fake, and the amplified difference, one column per method."""
    rng = np.random.default_rng(7)
    fakes = pd.concat([tests["aligned_vae"], tests["aligned_inpaint"]])
    cols = [("aligned_vae_sd15", "SD 1.5 VAE"), ("aligned_vae_sdxl", "SDXL VAE"), ("aligned_vae_flux", "FLUX.1 VAE"),
            ("aligned_vae_flux2", "FLUX.2 VAE"), ("aligned_vae_qwen", "Qwen-Image VAE"), ("aligned_inpaint", "SD2 inpainting")]
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.6))
    for c, (src, label) in enumerate(cols):
        cand = fakes[(fakes.source == src) & fakes.real.str.contains("/pexels/|hf_fairface|hf_of_real_pexels")]
        row = cand.iloc[rng.integers(len(cand))]
        fake = Image.open(DATA / row.path).convert("RGB")
        real_path = row.real if src != "aligned_inpaint" else row.path.replace("/inpaint/", "/real_resampled/")
        real = Image.open(DATA / real_path).convert("RGB").resize(fake.size)
        diff = np.clip(np.abs(np.asarray(real, float) - np.asarray(fake, float)) * 8, 0, 255).astype(np.uint8)
        for r, (im, name) in enumerate([(real, "real"), (fake, "fake"), (Image.fromarray(diff), "|difference| x 8")]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(label, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, color=INK)
    fig.suptitle("Content-matched fakes: same face, only generator traces differ", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "round_aligned.jpg", dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def results():
    row = json.loads((ROOT / "results" / "fm_combo.json").read_text())["best"]["row"]
    real = [("fairface", "FairFace"), ("gan_set_real", "StyleGAN dataset reals"), ("openfake_real", "OpenFake reals"),
            ("reddit_real", "Reddit photography"), ("aligned_real", "Reals behind aligned fakes"),
            ("photo_real", "Real photos"), ("video_real_frames", "Real video frames")]
    fake = [("openai", "OpenAI gpt-image"), ("sfhq", "SFHQ (FLUX/SDXL/DALL-E 3)"), ("stylegan", "StyleGAN"),
            ("rapidata_unseen", "Rapidata (never trained on)"), ("reddit_ai", "Reddit AI art"), ("dff_fake", "DeepFakeFace"),
            ("openfake", "OpenFake"), ("ai_video_frames", "AI video frames"), ("swap_frames", "Face-swap frames"),
            ("aligned_inpaint", "Aligned inpainting"), ("aligned_vae", "Aligned VAE round trip")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [7, 11]})
    for ax, items, color, title in ((a, real, GREEN, "Real sets: judged real"), (b, fake, RED, "AI sets: caught")):
        vals = [row[k] * 100 for k, _ in items]
        y = np.arange(len(items))[::-1]
        ax.barh(y, vals, color=color, height=0.62)
        ax.set_yticks(y, [n for _, n in items])
        ax.set_xlim(0, 100)
        for yi, v in zip(y, vals):
            ax.text(v - 1.5 if v > 15 else v + 1.5, yi, f"{v:.1f}%", va="center", ha="right" if v > 15 else "left",
                    color="white" if v > 15 else INK, fontsize=9, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("%")
    fig.suptitle("Held-out results, CLIP LoRA + DINOv2 ensemble (threshold: 97% of real photos judged real)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "round_results.png", dpi=110)
    plt.close(fig)


def calibration(tests, ens, thr):
    import calibrate as C

    c = json.loads((ROOT / "runs" / "fm_ens" / "calibration.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 1, 41)
    real = np.concatenate([ens[k] for k in C.REAL])
    fake = np.concatenate([ens[k] for k in C.FAKE])
    a.hist(real, bins, color=GREEN, alpha=.75, density=True, label="real sets")
    a.hist(fake, bins, color=RED, alpha=.6, density=True, label="AI sets")
    a.axvline(thr, color=INK, lw=1.2, ls="--")
    a.text(thr + .01, a.get_ylim()[1] * .9, f"threshold {thr}", fontsize=9, color=INK)
    a.set_yscale("log")
    a.set_xlabel("ensemble score")
    a.set_ylabel("density (log)")
    a.set_title("Score distribution on held-out sets", loc="left", fontsize=11, color=INK)
    a.legend(frameon=False)
    b.plot(c["x"], np.array(c["y"]) * 100, color=INK, lw=2)
    b.plot([0, 1], [0, 100], color=GREY, lw=1, ls=":")
    b.set_xlabel("ensemble score")
    b.set_ylabel("AI rate (%)")
    b.set_title(f"Score -> AI rate (isotonic, 50/50 prior; held-out ECE {c['ece_heldout']['raw']:.1%} -> "
                f"{c['ece_heldout']['isotonic']:.1%})", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "round_calibration.png", dpi=110)
    plt.close(fig)


def loss():
    pts = [(int(e), int(i), float(v)) for e, i, v in
           re.findall(r"^ep(\d+) (\d+)/\d+ loss ([\d.]+)", (ROOT / "logs" / "fm_ft_clip_lora.txt").read_text(), re.M)]
    per = max(i for _, i, _ in pts) + 34
    x = [e + i / per for e, i, _ in pts if i > 0]
    y = [v for _, i, v in pts if i > 0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, y, color=INK, lw=1.8, marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("running BCE loss")
    ax.set_title("CLIP ViT-L/14 LoRA training (real weight 2.5)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "round_loss.png", dpi=110)
    plt.close(fig)


def main():
    _, tests = F.data()
    tests.pop("my_photos", None)  # private, never shown
    ens, thr = ensemble_scores(tests)
    samples(tests, ens, thr)
    aligned(tests)
    results()
    calibration(tests, ens, thr)
    loss()


if __name__ == "__main__":
    main()
