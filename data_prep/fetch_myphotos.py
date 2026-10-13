"""Real faces from the owner's own Photos library: camera originals only, never edited.

  uvx osxphotos query --json --not-edited --only-photos > /tmp/myphotos_query.json
  uv run data_prep/fetch_myphotos.py /tmp/myphotos_query.json

Kept: not edited in Photos, not a screenshot, file on this Mac, camera make/model in EXIF, faces found
by Photos, and no editing software in the EXIF Software tag (camera firmware like "17.4" is fine).
Loose face crops (video_scan.faces_in, >= 224 px of real detail, at most 3 per photo) go to
data/img256/my_photos/ and data/myphotos_manifest.csv, split 80/20 by photo.
Private data: stays in data/ (gitignored); never uploaded, deployed or shown in public tables.
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from video_scan import faces_in

DATA = Path(__file__).resolve().parents[1] / "data"
EDITORS = re.compile(r"photoshop|lightroom|snapseed|meitu|美图|facetune|vsco|picsart|instagram|b612|faceu|"
                     r"beauty|retouch|gimp|pixelmator|affinity|canva|wechat|weibo|xingtu|醒图|remini", re.I)


def main(query):
    register_heif_opener()
    rows, skipped = [], {"edited/screenshot": 0, "no camera": 0, "no faces": 0, "missing": 0, "software": 0}
    for p in json.load(open(query)):
        exif = p.get("exif_info") or {}
        if p.get("hasadjustments") or p.get("screenshot"):
            skipped["edited/screenshot"] += 1
            continue
        if not exif.get("camera_make"):
            skipped["no camera"] += 1
            continue
        if not p.get("face_info"):
            skipped["no faces"] += 1
            continue
        path = Path(p.get("path") or "")
        if p.get("ismissing") or not path.exists():
            skipped["missing"] += 1
            continue
        try:
            with Image.open(path) as im:
                software = str(im.getexif().get(0x0131, ""))
                if EDITORS.search(software):
                    skipped["software"] += 1
                    continue
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            skipped["missing"] += 1
            continue
        found = [(b, c) for b, c in faces_in(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                 if 2.2 * max(b[2], b[3]) >= 224]
        uid = hashlib.md5(p["uuid"].encode()).hexdigest()[:16]  # no file names / dates in the manifest
        for k, (_, crop) in enumerate(sorted(found, key=lambda r: -r[0][2] * r[0][3])[:3]):
            rel = Path("img256") / "my_photos" / f"{uid}_{k}.jpg"
            (DATA / rel).parent.mkdir(parents=True, exist_ok=True)
            crop.save(DATA / rel, quality=95)
            rows.append({"path": str(rel), "source": "my_photos", "label": 0, "item": uid,
                         "split": "test" if int(uid, 16) % 5 == 0 else "train"})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "myphotos_manifest.csv", index=False)
    print("skipped", skipped)
    print(df.groupby("split").agg(crops=("path", "size"), photos=("item", "nunique")).to_string() if len(df) else "no crops")


if __name__ == "__main__":
    main(sys.argv[1])
