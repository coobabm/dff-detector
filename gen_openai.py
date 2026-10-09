"""Generate face photos with an OpenAI-compatible image API for out-of-distribution testing.

  uv run gen_openai.py --n 100 --model gpt-image-2

Key: env OPENAI_API_KEY or file openai.key (gitignored).
Endpoint: env OPENAI_BASE_URL (default https://api.openai.com/v1). Originals go to data/openai/orig,
256x256 JPEGs (same preprocessing as DFF) to data/img256/openai.
"""
import argparse
import base64
import os
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).parent
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
KEY = os.environ.get("OPENAI_API_KEY") or (ROOT / "openai.key").read_text().strip()

AGES = ["young", "20-year-old", "30-year-old", "middle-aged", "50-year-old", "elderly"]
GENDERS = ["man", "woman"]
ETHNIC = ["East Asian", "South Asian", "Black", "white", "Latino", "Middle Eastern"]
SCENES = [
    "at a red carpet event, paparazzi flash",
    "in a press conference",
    "outdoors in daylight",
    "in an office",
    "on a street, candid snapshot",
    "at a sports event",
    "in a TV studio",
    "at home, indoor lighting",
]
STYLES = [
    "photo taken with a digital camera in 2008",
    "news agency photograph",
    "amateur phone photo",
    "professional headshot",
    "old scanned photograph",
]


def prompt(rng):
    return (f"Realistic photograph of a {rng.choice(AGES)} {rng.choice(ETHNIC)} {rng.choice(GENDERS)} "
            f"{rng.choice(SCENES)}, face clearly visible, upper body, {rng.choice(STYLES)}. "
            "No text, no watermark.")


def square256(im):
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((256, 256), Image.BICUBIC)


def gen(job):
    i, p, model, orig, out = job
    dst = out / f"{i:04d}.jpg"
    if dst.exists():
        return i, "skip"
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/images/generations", timeout=300,
                              headers={"Authorization": f"Bearer {KEY}"},
                              json={"model": model, "prompt": p, "n": 1, "size": "1024x1024"})
            r.raise_for_status()
            raw = base64.b64decode(r.json()["data"][0]["b64_json"])
            break
        except Exception as e:
            err = str(e)[:200]
    else:
        return i, f"fail {err}"
    (orig / f"{i:04d}.png").write_bytes(raw)
    square256(Image.open(io.BytesIO(raw)).convert("RGB")).save(dst, quality=95)
    return i, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=["gpt-image-2"], help="rotated per image")
    ap.add_argument("--start", type=int, default=0, help="first index that uses model rotation")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    orig = ROOT / "data" / "openai" / "orig"
    out = ROOT / "data" / "img256" / "openai"
    orig.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    prompts = [prompt(rng) for _ in range(args.n)]
    json.dump(prompts, open(orig.parent / "prompts.json", "w"), ensure_ascii=False, indent=1)
    model_of = {i: (args.models[0] if i < args.start else args.models[i % len(args.models)]) for i in range(args.n)}
    meta_path = orig.parent / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    jobs = [(i, p, model_of[i], orig, out) for i, p in enumerate(prompts)]
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, status in ex.map(gen, jobs):
            print(i, model_of[i], status, flush=True)
            if status == "ok":
                meta[f"{i:04d}"] = {"model": model_of[i], "prompt": prompts[i]}
                json.dump(meta, open(meta_path, "w"), indent=1)
            fails = fails + 1 if status.startswith("fail") else 0
            if fails >= 20:
                print("20 consecutive failures, stopping (quota?)", flush=True)
                ex.shutdown(cancel_futures=True)
                break


if __name__ == "__main__":
    main()
