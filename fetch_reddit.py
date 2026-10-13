"""Collect AI-generated portraits from AI-art subreddits (Midjourney, Flux, SD, GPT image, Gemini ...).

  uv run fetch_reddit.py list      # search posts via rdt-cli (needs `rdt login`), write data/reddit/posts.jsonl
  uv run fetch_reddit.py list_real # real photos from photography subs -> data/reddit/posts_real.jsonl
  uv run fetch_reddit.py download  # download the images (i.redd.it originals)
  uv run fetch_reddit.py faces     # loose face crops (same crop as video frames / app.py) -> img256/rd_<gen>/

AI side: only AI-art subreddits; posts whose titles suggest real photos ("real", "photo of me" ...) are
skipped. Real side: public photography subreddits (portrait, phone and street photography); NSFW posts and
posts mentioning AI tools are skipped. No usernames are stored. Images stay local (data/ is gitignored)
and are not redistributed.
The split is by post, so crops from one post never land in both train and test.
"""
import hashlib
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "reddit"
QUERIES = ["portrait", "woman", "man", "face", "realistic", "photorealistic", "selfie", "close up", "girl", "old man"]
SUBS = {  # subreddit -> generator label
    "midjourney": "midjourney",
    "FluxAI": "flux",
    "StableDiffusion": "sd_mixed",
    "dalle2": "dalle",
    "ChatGPT": "gpt_image",
    "OpenAI": "gpt_image",
    "GeminiAI": "gemini",
    "Bard": "gemini",
    "nanobanana": "gemini",
    "ideogramai": "ideogram",
    "leonardoai": "leonardo",
    "aiArt": "ai_mixed",
    "AIArtwork": "ai_mixed",
    "SeaArtAI": "ai_mixed",
}
REAL_SUBS = {  # subreddit -> real source label (photos people posted to share)
    "portraits": "real_portrait",
    "portraitphotography": "real_portrait",
    "itookapicture": "real_photo",
    "iphonephotography": "real_phone",
    "ShotOnIphone": "real_phone",
    "pixelphotography": "real_phone",
    "AndroidPhotography": "real_phone",
    "streetphotography": "real_street",
    "PortraitPhotos": "real_portrait",
}
REAL_QUERIES = ["portrait", "face", "selfie", "friend", "candid", "headshot", "smile"]
AI_WORDS = re.compile(r"\bai\b|a\.i\.|midjourney|generat|stable ?diffusion|flux|dall|gpt|gemini|nano ?banana|"
                      r"render|cgi|3d|illustration|painting|drawing", re.I)
# ChatGPT / OpenAI / Gemini are general subs: only keep posts that talk about generating images.
GENERAL = {"ChatGPT", "OpenAI", "GeminiAI", "Bard"}
IMAGE_WORDS = re.compile(r"image|generat|portrait|photo|picture|4o|gpt-image|imagen|nano ?banana|sora", re.I)
REALISH = re.compile(r"\breal\b|\bvs\.? real|photo of me|\bmy (wife|husband|gf|bf|son|daughter|mom|dad)\b|"
                     r"\bme\b.*\bvs\b|before.*after|which (one )?is (ai|real)", re.I)
UA = {"User-Agent": "Mozilla/5.0 (research image collection for an AI-face detector)"}


def rdt(args):
    for attempt in range(3):
        r = subprocess.run(["rdt", *args, "--json"], capture_output=True, text=True, timeout=120)
        try:
            d = json.loads(r.stdout)
            data = d["data"]
            return data["data"] if "children" not in data else data
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def image_urls(p):
    """Original-resolution image URLs of a post (single image or gallery)."""
    urls = []
    if p.get("is_gallery") and p.get("media_metadata"):
        order = [g["media_id"] for g in (p.get("gallery_data") or {}).get("items", [])] or list(p["media_metadata"])
        for mid in order:
            m = p["media_metadata"].get(mid, {})
            if m.get("status") == "valid" and m.get("e") == "Image":
                ext = m.get("m", "image/jpg").split("/")[-1].replace("jpeg", "jpg")
                urls.append(f"https://i.redd.it/{mid}.{ext}")
    elif re.search(r"\.(jpe?g|png|webp)$", p.get("url", ""), re.I) and "i.redd.it" in p.get("url", ""):
        urls.append(p["url"])
    return [html.unescape(u) for u in urls][:6]


def list_real(pages=3):
    return list_posts(pages, real=True)


def list_posts(pages=3, real=False):
    OUT.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    path = OUT / ("posts_real.jsonl" if real else "posts.jsonl")
    subs, queries = (REAL_SUBS, REAL_QUERIES) if real else (SUBS, QUERIES)
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            seen.add(r["id"])
            rows.append(r)
    with path.open("a") as f:
        for sub, gen in subs.items():
            for q in queries:
                for sort, t in (("top", "all"), ("relevance", "year")):
                    after, kept = None, 0
                    for _ in range(pages):
                        args = ["search", q, "-r", sub, "--sort", sort, "--time", t, "-n", "100"]
                        L = rdt(args + (["--after", after] if after else []))
                        if not L:
                            break
                        for c in L.get("children", []):
                            p = c["data"]
                            title = p.get("title", "")
                            if p["id"] in seen or p.get("over_18"):
                                continue
                            if (AI_WORDS if real else REALISH).search(title + " " + (p.get("link_flair_text") or "")):
                                continue
                            if sub in GENERAL and not IMAGE_WORDS.search(title + " " + (p.get("link_flair_text") or "")):
                                continue
                            urls = image_urls(p)
                            if not urls:
                                continue
                            seen.add(p["id"])
                            row = {"id": p["id"], "sub": sub, "gen": gen, "title": title,
                                   "flair": p.get("link_flair_text"), "created": p.get("created_utc"), "urls": urls}
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)
                            kept += 1
                        after = L.get("after")
                        time.sleep(1.5)
                        if not after:
                            break
                    print(f"{sub:16s} {q:15s} {sort:9s} +{kept}  total {len(rows)}", flush=True)
    return rows


def download(workers=8):
    from concurrent.futures import ThreadPoolExecutor

    rows = [json.loads(l) for name in ("posts.jsonl", "posts_real.jsonl") if (OUT / name).exists()
            for l in (OUT / name).open()]
    jobs = [(u, OUT / "raw" / r["gen"] / f"{r['id']}_{i}{Path(u).suffix.lower()}")
            for r in rows for i, u in enumerate(r["urls"])]
    jobs = [j for j in jobs if not j[1].exists()]
    s = requests.Session()

    def get(job):
        u, dst = job
        try:
            resp = s.get(u, headers=UA, timeout=30)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image") \
                    and len(resp.content) < 25e6:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(resp.content)
                return 1
        except requests.RequestException:
            pass
        return 0

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for k, ok in enumerate(ex.map(get, jobs), 1):
            n += ok
            if k % 500 == 0:
                print(f"{k}/{len(jobs)} tried, {n} saved", flush=True)
    print(f"{len(jobs)} tried, {n} saved")


def faces():
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    from video_scan import faces_in

    register_heif_opener()
    rows, dup = [], set()
    for f in sorted((OUT / "raw").glob("*/*")):
        gen, post = f.parent.name, f.stem.split("_")[0]
        try:
            with Image.open(f) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            continue
        digest = hashlib.md5(img.resize((32, 32)).tobytes()).hexdigest()  # reposts / cross-posts
        if digest in dup:
            continue
        dup.add(digest)
        found = faces_in(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        # Only faces with enough real pixels (same rule as app.face_crop), at most 3 per image.
        found = [(b, c) for b, c in found if 2.2 * max(b[2], b[3]) >= 224]
        found = sorted(found, key=lambda r: -r[0][2] * r[0][3])[:3]
        for k, (_, crop) in enumerate(found):
            rel = Path("img256") / f"rd_{gen}" / f"{f.stem}_{k}.jpg"
            (DATA / rel).parent.mkdir(parents=True, exist_ok=True)
            crop.save(DATA / rel, quality=95)
            rows.append({"path": str(rel), "source": f"rd_{gen}", "label": 0 if gen.startswith("real_") else 1,
                         "post": post})
    df = pd.DataFrame(rows)
    # 20% of posts held out for testing.
    df["split"] = df.post.map(lambda p: "test" if int(hashlib.md5(p.encode()).hexdigest(), 16) % 5 == 0 else "train")
    df.to_csv(DATA / "reddit_manifest.csv", index=False)
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    {"list": list_posts, "list_real": list_real, "download": download, "faces": faces}[sys.argv[1]]()
