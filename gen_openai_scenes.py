"""Generate everyday scene photos (not just portraits) with an OpenAI-compatible image API.

  OPENAI_BASE_URL=... uv run gen_openai_scenes.py [--per 50] [--model gpt-image-2]

5 categories x --per images, each prompt built from random parts so no two are alike:
street, indoor, travel, night, phone snapshot. Aspect ratio varies like real photos.
Key: env OPENAI_API_KEY or openai.key (gitignored); endpoint from env OPENAI_BASE_URL.
Originals -> data/openai_scenes/<category>/NNNN.png, prompts -> data/openai_scenes/meta.json.
"""
import argparse
import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "openai_scenes"
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
KEY = os.environ.get("OPENAI_API_KEY") or (ROOT / "openai.key").read_text().strip()

PLACES = ["Tokyo", "Seoul", "Shanghai", "New York", "Paris", "Mexico City", "Mumbai", "Lagos", "Istanbul",
          "London", "Bangkok", "São Paulo", "a small European town", "a Midwest American suburb"]
WEATHER = ["sunny afternoon", "overcast day", "light rain", "golden hour", "early morning fog", "snowy day"]
CAMERAS = ["shot on an iPhone", "shot on an Android phone", "35mm film photo", "DSLR photo, 50mm lens",
           "point-and-shoot camera photo", "slightly blurry casual snapshot"]
CATEGORIES = {
    "street": lambda r: (f"Candid street photograph in {r.choice(PLACES)} on a {r.choice(WEATHER)}, "
                         f"{r.choice(['pedestrians crossing', 'a busy market', 'people waiting at a bus stop', 'a cyclist passing shops', 'a street food stall with customers', 'commuters outside a subway entrance'])}, "
                         f"{r.choice(CAMERAS)}"),
    "indoor": lambda r: (f"Everyday indoor photo of {r.choice(['friends chatting in a cafe', 'a family cooking in a small kitchen', 'coworkers in an open office', 'students in a library', 'people at a birthday party', 'a barber cutting hair', 'a living room with a person on the sofa'])}, "
                         f"{r.choice(['warm lamp light', 'fluorescent office light', 'window daylight', 'mixed indoor lighting'])}, "
                         f"{r.choice(CAMERAS)}"),
    "travel": lambda r: (f"Travel photo of {r.choice(['a mountain trail with hikers', 'a beach with a few swimmers', 'tourists at an old temple', 'a lake with a wooden pier', 'a desert road with a parked car', 'a countryside village', 'a waterfall viewpoint with visitors'])} "
                         f"near {r.choice(PLACES)}, {r.choice(WEATHER)}, {r.choice(CAMERAS)}"),
    "night": lambda r: (f"Night photo in {r.choice(PLACES)}: {r.choice(['neon signs above a crowded alley', 'a concert crowd under stage lights', 'a quiet convenience store at 2am', 'a night market', 'car headlights on a wet road', 'people outside a bar'])}, "
                        f"{r.choice(['high ISO noise', 'motion blur from handheld shot', 'flash photo', 'long exposure'])}, {r.choice(CAMERAS)}"),
    "phone": lambda r: (f"Casual phone snapshot posted on social media: {r.choice(['a plate of food on a restaurant table', 'a dog on a couch', 'a cat by the window', 'a group selfie of friends', 'a mirror selfie in a bedroom', 'a messy desk with a laptop', 'a child blowing out birthday candles', 'coffee and a book at a cafe'])}, "
                        f"{r.choice(['slightly tilted', 'auto HDR look', 'warm filter', 'no filter', 'portrait mode background blur'])}, {r.choice(CAMERAS[:2])}"),
}
SIZES = ["1024x1024", "1536x1024", "1024x1536"]


def gen(job):
    cat, i, prompt, size, model = job
    dst = OUT / cat / f"{i:04d}.png"
    if dst.exists():
        return job, "skip"
    err = ""
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/images/generations", timeout=300, headers={"Authorization": f"Bearer {KEY}"},
                              json={"model": model, "prompt": prompt + ". No text, no watermark.", "n": 1, "size": size})
            r.raise_for_status()
            dst.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
            return job, "ok"
        except Exception as e:
            err = str(e)[:200]
    return job, "fail " + err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--model", default="gpt-image-2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(0)
    jobs = []
    for cat, make in CATEGORIES.items():
        (OUT / cat).mkdir(parents=True, exist_ok=True)
        jobs += [(cat, i, make(rng), rng.choice(SIZES), args.model) for i in range(args.per)]
    meta_path = OUT / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    ok = fail = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for (cat, i, prompt, size, model), status in ex.map(gen, jobs):
            if status in ("ok", "skip"):
                ok += 1
                meta[f"{cat}/{i:04d}"] = {"model": model, "size": size, "prompt": prompt}
                json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
            else:
                fail += 1
                print(cat, i, status, flush=True)
            if (ok + fail) % 25 == 0:
                print(f"{ok + fail}/{len(jobs)} done, {fail} failed", flush=True)
    print(f"finished: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
