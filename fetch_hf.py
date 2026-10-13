"""Modern-generator faces and varied real faces from Hugging Face datasets.

  uv run fetch_hf.py sfhq       # SFHQ-T2I (bitmind mirror, MIT): FLUX pro/dev/schnell, SDXL, DALL-E 3 portraits
  uv run fetch_hf.py openfake   # OpenFake core (CC-BY-NC-4.0): gpt-image, nano-banana, Imagen, MJ, Seedream ... + LAION/Pexels reals
  uv run fetch_hf.py fairface   # FairFace 1.25 (CC-BY-4.0): real faces, all ages / ethnicities
  uv run fetch_hf.py rapidata   # Rapidata face benchmark (CDLA-permissive-2.0): newest generators, test only
  uv run fetch_hf.py manifest   # -> data/hf_manifest.csv

Every image goes through the same loose face crop as the rest of the project (video_scan.faces_in,
face crop >= 224 px of real detail, at most 2 faces), saved as 256 px JPEG q95 under img256/hf_<source>/.
Real and fake therefore share one resolution / format pipeline. 20% of items are held out (by item hash);
Rapidata is test-only. Nothing here is redistributed.
"""
import hashlib
import io
import random
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from video_scan import faces_in

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HF = "https://huggingface.co/datasets"
PERSON = re.compile(r"\b(portrait|face|selfie|headshot|man|woman|girl|boy|person|people|lady|guy|child|kid|"
                    r"teen|elderly|couple|family|businessman|student|model)\b", re.I)


def slug(s):
    return re.sub(r"[^a-z0-9.]+", "_", str(s).lower()).strip("_")


def crops(img, k=2):
    img = ImageOps.exif_transpose(img).convert("RGB")
    found = [(b, c) for b, c in faces_in(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
             if 2.2 * max(b[2], b[3]) >= 224]
    return [c for _, c in sorted(found, key=lambda r: -r[0][2] * r[0][3])[:k]]


def save(img, source, item, label, rows, k=2):
    for j, c in enumerate(crops(img, k)):
        rel = Path("img256") / f"hf_{source}" / f"{slug(item)[:80]}_{j}.jpg"
        (DATA / rel).parent.mkdir(parents=True, exist_ok=True)
        c.save(DATA / rel, quality=95)
        rows.append({"path": str(rel), "source": f"hf_{source}", "label": label, "item": str(item)})


def write_rows(name, rows):
    out = DATA / "hf_parts"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(name, pd.DataFrame(rows).groupby(["source", "label"]).size().to_string() if rows else "nothing", flush=True)


def sfhq(per_big=2000, workers=6):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import fsspec

    url = f"{HF}/bitmind/SyntheticFacesHighQuality-T2I/resolve/main/SyntheticFacesHighQuality.zip"
    for attempt in range(10):
        try:
            z = zipfile.ZipFile(fsspec.open(url, block_size=1 << 18).open())
            break
        except Exception:
            import time

            time.sleep(30 * (attempt + 1))
    files = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith((".jpg", ".png"))]
    groups = {}
    for i in files:
        groups.setdefault(Path(i.filename).name.split("_image_")[0], []).append(i)
    print({k: len(v) for k, v in groups.items()}, flush=True)
    rng = random.Random(0)
    local = threading.local()

    def fetch(info):  # one remote zip handle per thread; each file is an HTTP range read
        import time

        for attempt in range(6):  # Hugging Face rate-limits unauthenticated range requests: back off and reopen
            try:
                if not hasattr(local, "z"):
                    local.z = zipfile.ZipFile(fsspec.open(url, block_size=1 << 18).open())
                return info, local.z.read(info)
            except Exception:
                local.__dict__.pop("z", None)
                time.sleep(10 * (attempt + 1))
        return info, None

    rows = []
    for model, items in groups.items():
        pick = items if len(items) <= 12000 and model not in ("FLUX1_schnell", "SDXL") else rng.sample(items, per_big)
        src = f"sfhq_{slug(model)}"
        pick = [i for i in pick if not (DATA / "img256" / f"hf_{src}" / f"{slug(Path(i.filename).stem)[:80]}_0.jpg").exists()]
        for n, (info, raw) in enumerate(ThreadPoolExecutor(workers).map(fetch, pick)):
            if raw is not None:
                with Image.open(io.BytesIO(raw)) as im:  # face detection stays on this thread (YuNet is not thread-safe)
                    save(im, src, Path(info.filename).stem, 1, rows, k=1)
            if n % 500 == 0:
                print("sfhq", model, n, "/", len(pick), flush=True)
    # Rows of crops saved by an earlier, interrupted run.
    for d in (DATA / "img256").glob("hf_sfhq_*"):
        have = {r["path"] for r in rows}
        rows += [{"path": str(f.relative_to(DATA)), "source": d.name, "label": 1, "item": f.stem.rsplit("_", 1)[0]}
                 for f in d.glob("*.jpg") if str(f.relative_to(DATA)) not in have]
    write_rows("sfhq", rows)


def openfake(shards=("core/train-00000-of-00032-00000.parquet", "core/train-00011-of-00032-00002.parquet",
                     "core/train-00023-of-00032-00004.parquet"), name="openfake"):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = []
    for shard in shards:
        local = hf_hub_download("ComplexDataLab/OpenFake", shard, repo_type="dataset",
                                local_dir=DATA / "hf" / "openfake")
        pf = pq.ParquetFile(local)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["image", "prompt", "label", "model", "image_sha256"]).to_pylist()
            for r in t:
                if not PERSON.search(r["prompt"] or ""):
                    continue
                real = str(r["label"]).lower() in ("real", "0")
                src = ("of_real_" if real else "of_") + slug(r["model"])
                raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        save(im, src, r["image_sha256"] or hashlib.md5(raw).hexdigest(), 0 if real else 1, rows)
                except Exception:
                    continue
            if g % 10 == 0:
                print(shard, "row group", g, "/", pf.metadata.num_row_groups, "crops", len(rows), flush=True)
        Path(local).unlink()  # keep only the crops
    write_rows(name, rows)


def parquet_images(repo, files, image_cols, meta=lambda r, c: c):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for f in files:
        pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset", local_dir=DATA / "hf" / slug(repo)))
        for g in range(pf.metadata.num_row_groups):
            for k, r in enumerate(pf.read_row_group(g).to_pylist()):
                for c in image_cols:
                    img = r[c]
                    raw = img["bytes"] if isinstance(img, dict) else img
                    yield f"{Path(f).stem}_{g}_{k}_{c}", meta(r, c), raw


def fairface():
    rows = []
    files = ["1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet", "1.25/train-00000-of-00004-e715178553977907.parquet"]
    for n, (item, _, raw) in enumerate(parquet_images("HuggingFaceM4/FairFace", files, ["image"])):
        with Image.open(io.BytesIO(raw)) as im:
            save(im, "fairface", item, 0, rows, k=1)
        if n % 2000 == 0:
            print("fairface", n, len(rows), flush=True)
    write_rows("fairface", rows)


def rapidata():
    rows, seen = [], set()
    files = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]
    for item, model, raw in parquet_images("Rapidata/Face_Generation_Benchmark", files, ["image1", "image2"],
                                           meta=lambda r, c: r["model1" if c == "image1" else "model2"]):
        h = hashlib.md5(raw).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        with Image.open(io.BytesIO(raw)) as im:
            save(im, "rapidata_" + slug(model), h, 1, rows, k=1)
    write_rows("rapidata", rows)


def manifest():
    df = pd.concat([pd.read_csv(f) for f in sorted((DATA / "hf_parts").glob("*.csv"))], ignore_index=True)
    held = df.item.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 5 == 0)
    df["split"] = np.where(df.source.str.startswith("hf_rapidata") | held, "test", "train")
    df.to_csv(DATA / "hf_manifest.csv", index=False)
    print(df.groupby(["source", "label", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "openfake" and len(sys.argv) > 2:  # extra shards: fetch_hf.py openfake <list file> <part name>
        openfake(tuple(Path(sys.argv[2]).read_text().split()), sys.argv[3])
    else:
        {"sfhq": sfhq, "openfake": openfake, "fairface": fairface, "rapidata": rapidata, "manifest": manifest}[sys.argv[1]]()
