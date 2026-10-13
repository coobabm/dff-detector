"""Download modern real-people photos to pair with OpenAI fakes.

  uv run data_prep/fetch_real.py ffhq pexels ppr10k flickr30k easyportrait

Each source -> data/img256/<source>/*.jpg (center square crop, 256x256, JPEG q95,
same preprocessing as the OpenAI images).
"""
import csv
import gzip
import io
import random
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "realsrc"
N = 3000
HF = "https://huggingface.co/datasets"


def square256(im):
    im = im.convert("RGB")
    w, h = im.size
    s = min(w, h)
    # Portraits: bias crop upward so the face stays in frame.
    top = 0 if h > w else (h - s) // 2
    im = im.crop(((w - s) // 2, top, (w - s) // 2 + s, top + s))
    return im.resize((256, 256), Image.BICUBIC)


def save(raw, dst):
    try:
        square256(Image.open(io.BytesIO(raw))).save(dst, quality=95)
        return True
    except Exception as e:
        print("bad", dst.name, e, file=sys.stderr)
        return False


def fetch_many(items, out):
    """items: list of (name, url)."""
    out.mkdir(parents=True, exist_ok=True)

    def one(item):
        name, url = item
        dst = out / f"{name}.jpg"
        if dst.exists():
            return True
        for _ in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                return save(r.content, dst)
            except Exception as e:
                err = e
        print("fail", name, err, file=sys.stderr)
        return False

    with ThreadPoolExecutor(16) as ex:
        ok = sum(ex.map(one, items))
    print(out.name, "ok", ok, "/", len(items), flush=True)


def pexels():
    rows = list(csv.DictReader(gzip.open(SRC / "pexels.csv.gz", "rt")))
    # Pexels ids below ~14M are uploads from before 2023, i.e. before photoreal AI images were common.
    rows = [r for r in rows if int(r["id"]) < 14_000_000]
    random.Random(0).shuffle(rows)
    items = [(r["id"], r["src"] + "?auto=compress&cs=tinysrgb&w=800") for r in rows[:N]]
    print("pexels candidates", len(rows))
    fetch_many(items, DATA / "img256" / "pexels")


def ppr10k():
    files = requests.get("https://huggingface.co/api/datasets/JarvisArt/MMArt-PPR10k").json()["siblings"]
    names = sorted(f["rfilename"] for f in files if f["rfilename"].endswith("before.jpg"))
    random.Random(0).shuffle(names)
    items = [(n.split("/")[1], f"{HF}/JarvisArt/MMArt-PPR10k/resolve/main/{n}") for n in names[:N]]
    fetch_many(items, DATA / "img256" / "ppr10k")


def from_zip(name, repo, member, filt=lambda n: True):
    zp = SRC / f"{name}.zip"
    done = zp.with_suffix(".done")
    if not done.exists():
        subprocess.run(["curl", "-sL", "--retry", "10", "--retry-all-errors", "--http1.1", "-C", "-", "-o", str(zp), f"{HF}/{repo}/resolve/main/{member}"], check=True)
        done.touch()
    out = DATA / "img256" / name
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and filt(n))
        random.Random(0).shuffle(names)
        ok = 0
        for n in names[:N]:
            dst = out / (Path(n).stem + ".jpg")
            ok += dst.exists() or save(zf.read(n), dst)
    print(name, "ok", ok, "/", min(N, len(names)), flush=True)
    zp.unlink()
    done.unlink()


def ffhq():
    """First shard of bitmind/ffhq-256 (4,375 aligned Flickr faces, already 256x256)."""
    import pyarrow.parquet as pq

    pq_path = SRC / "ffhq_shard0.parquet"
    if not pq_path.with_suffix(".done").exists():
        subprocess.run(["curl", "-sL", "--retry", "10", "-C", "-", "-o", str(pq_path),
                        f"{HF}/bitmind/ffhq-256/resolve/main/data/train-00000-of-00016.parquet"], check=True)
        pq_path.with_suffix(".done").touch()
    out = DATA / "img256" / "ffhq"
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(pq_path).column("image").to_pylist()
    for i, r in enumerate(rows):
        dst = out / f"{i:05d}.jpg"
        if not dst.exists():
            Image.open(io.BytesIO(r["bytes"])).convert("RGB").resize((256, 256), Image.BICUBIC).save(dst, quality=95)
    print("ffhq ok", len(rows), flush=True)
    pq_path.unlink()
    pq_path.with_suffix(".done").unlink()


def flickr30k():
    from_zip("flickr30k", "nlphuji/flickr30k", "flickr30k-images.zip")


def easyportrait():
    from_zip("easyportrait", "gofixyourself/EasyPortrait", "data/test.zip")


if __name__ == "__main__":
    SRC.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:]:
        globals()[s]()
