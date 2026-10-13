"""Download StyleGAN face images (and matching real FFHQ faces) as 256x256 JPEGs.

  uv run data_prep/fetch_gan.py

Sources (fixed seed, ~1,000-1,500 each):
  tpdne     javi22/this-person-does-not-exist-10k   StyleGAN2 (thispersondoesnotexist.com)
  sg2face   gojay/StyleGAN2-Face                    StyleGAN2
  sg3       34data/StyleGAN3_1024x1024_images        StyleGAN3
  k140_fake TheKernel01/140k-Real-and-Fake-Faces    StyleGAN (fake half)
  k140_real TheKernel01/140k-Real-and-Fake-Faces    FFHQ (real half)
Output: data/img256/gan/<source>/*.jpg, manifest data/gan_manifest.csv
(path, source, label, split; 20% of each source is test).
"""
import csv
import io
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import HfApi
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "img256" / "gan"
HF = "https://huggingface.co/datasets"
rng = random.Random(0)


def to256(raw):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((256, 256), Image.BICUBIC)


def save(source, name, raw, label, rows):
    dst = OUT / source / f"{Path(name).stem.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not dst.exists():
            to256(raw).save(dst, quality=95)
    except Exception as e:
        print("  bad", name, e, flush=True)
        return
    split = "test" if random.Random(f"{source}/{name}").random() < 0.2 else "train"
    rows.append([str(dst.relative_to(DATA)), source, label, split])


def from_files(source, repo, n, rows, label=1):
    files = sorted(f.path for f in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)
                   if f.path.lower().endswith((".jpg", ".jpeg", ".png")))
    rng.shuffle(files)

    def get(path):
        for _ in range(3):
            try:
                r = requests.get(f"{HF}/{repo}/resolve/main/{path}", timeout=120)
                r.raise_for_status()
                return path, r.content
            except Exception:
                pass
        return path, None

    with ThreadPoolExecutor(16) as ex:
        for path, raw in ex.map(get, files[:n]):
            if raw:
                save(source, path.replace("/", "_"), raw, label, rows)
    print(source, len([r for r in rows if r[1] == source]), flush=True)


def from_parquet(repo, member, picks, rows):
    """picks: list of (source, label_filter or None, n, label)."""
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.parquet"
        subprocess.run(["curl", "-sL", "--retry", "10", "-o", str(p), f"{HF}/{repo}/resolve/main/{member}"], check=True)
        t = pq.read_table(p).to_pandas()
    img_col = next(c for c in t.columns if "image" in c.lower())
    for source, filt, n, label in picks:
        sub = t if filt is None else t[filt(t)]
        sub = sub.sample(min(n, len(sub)), random_state=0)
        for i, (_, row) in enumerate(sub.iterrows()):
            cell = row[img_col]
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            save(source, f"{source}_{i:05d}", raw, label, rows)
        print(source, len([r for r in rows if r[1] == source]), flush=True)


def main():
    rows = []
    from_files("tpdne", "javi22/this-person-does-not-exist-10k", 1500, rows)
    from_files("sg2face", "gojay/StyleGAN2-Face", 1000, rows)
    from_parquet("34data/StyleGAN3_1024x1024_images", "data.parquet", [("sg3", None, 1000, 1)], rows)
    # 140k: label column marks real (FFHQ) vs fake (StyleGAN).
    lab = lambda t: t[next(c for c in t.columns if c.lower() in ("label", "labels"))]
    from_parquet("TheKernel01/140k-Real-and-Fake-Faces", "data/test-00000-of-00002.parquet",
                 [("k140_fake", lambda t: lab(t).astype(str).str.lower().isin(["1", "fake"]), 1000, 1),
                  ("k140_real", lambda t: lab(t).astype(str).str.lower().isin(["0", "real"]), 1000, 0)], rows)
    with open(DATA / "gan_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "split"])
        w.writerows(rows)
    print("total", len(rows))


if __name__ == "__main__":
    main()
