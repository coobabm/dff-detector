"""Unzip DeepFakeFace, resize every image to 256x256, write split manifest.

Split is by IMDB-WIKI subfolder (person id mod 100), so one person never
appears in two splits. Folders 00-79 train, 80-89 val, 90-99 test.
"""
import csv
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent / "data"
RAW = ROOT / "raw"
OUT = ROOT / "img256"
SOURCES = {"wiki": 0, "text2img": 1, "inpainting": 1, "insight": 1}
SIZE = 256


def split_of(folder: int) -> str:
    if folder < 80:
        return "train"
    if folder < 90:
        return "val"
    return "test"


def process(args):
    zip_path, names = args
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in names:
            dst = OUT / name
            parts = Path(name).parts  # source/folder/file.jpg
            row = (str(dst.relative_to(ROOT)), parts[0], SOURCES[parts[0]], split_of(int(parts[1])))
            if dst.exists():
                rows.append(row)
                continue
            try:
                with zf.open(name) as f:
                    im = Image.open(f).convert("RGB")
                    im = im.resize((SIZE, SIZE), Image.BICUBIC)
            except Exception as e:  # corrupt file
                print("skip", name, e, file=sys.stderr)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=95)
            rows.append(row)
    return rows


def main():
    jobs = []
    for src in SOURCES:
        zp = RAW / f"{src}.zip"
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".png", ".jpeg"))]
        for i in range(0, len(names), 500):
            jobs.append((zp, names[i : i + 500]))
    rows = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(process, jobs)):
            rows.extend(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} chunks", flush=True)
    with open(ROOT / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "split"])
        w.writerows(sorted(rows))
    print("images:", len(rows))


if __name__ == "__main__":
    main()
