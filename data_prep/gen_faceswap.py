"""Make FaceFusion face-swap videos from real CelebV-HQ clips, then extract face frames.

  uv run data_prep/gen_faceswap.py --clips 150 --swaps-per-clip 2                       # round 3
  uv run data_prep/gen_faceswap.py --clips 120 --targets raw --faces ffhq --tag ffusion2  # round 4

Source faces: OpenAI-generated (people who do not exist) or real FFHQ photos. For each real clip:
  - the original and the swaps are both encoded once from the raw clip with the same
    libx264 quality, so
    "extra compression" cannot be used as a shortcut;
  - N swaps are made with rotating FaceFusion models and pixel-boost settings.
Frames (same frame indices for real and swaps) go to data/img256/video/ffusion_*,
manifest to data/video_manifest/facefusion.csv. Split is by YouTube id.
Needs FaceFusion 3.x at ~/Applications/facefusion (path via FACEFUSION_DIR).
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import argparse
import csv
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi

from fetch_video import DATA, HF, frames_to_faces, split_of

ROOT = Path(__file__).resolve().parents[1]
FF = Path(os.environ.get("FACEFUSION_DIR", Path.home() / "Applications" / "facefusion"))
MODELS = ["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "alphaface_256", "inswapper_128"]
BOOSTS = ["256x256", "512x512"]


def ff_quality_to_crf(q):
    # FaceFusion maps output-video-quality to libx264 crf this way.
    return round(51 - q * 0.51)


def swap(src_img, target, out, model, boost, quality):
    cmd = [str(FF / ".venv" / "bin" / "python"), "facefusion.py", "headless-run",
           "-s", str(src_img), "-t", str(target), "-o", str(out),
           "--processors", "face_swapper", "--face-swapper-model", model,
           "--face-swapper-pixel-boost", boost,
           "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_3",
           "--face-selector-mode", "one", "--execution-providers", "coreml",
           "--output-video-encoder", "libx264", "--output-video-quality", str(quality)]
    r = subprocess.run(cmd, cwd=FF, capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


def keep(video, tag, kind, name, label, group, split):
    dst = DATA / "videos" / tag / kind / f"{name}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(video, dst)
    return [str(dst.relative_to(DATA)), f"{tag}_{kind}", label, f"celebv/{group}", split]


def clip_targets(rng, n):
    """Short real CelebV-HQ talking-head clips."""
    clips = sorted(f.path for f in HfApi().list_repo_tree("saeed-5959/celebv_hq_head_talking", repo_type="dataset",
                                                         recursive=True) if f.path.endswith(".mp4"))
    rng.shuffle(clips)
    for clip in clips[:n]:
        stem = Path(clip).stem

        def fetch(dst, clip=clip):
            dst.write_bytes(requests.get(f"{HF}/saeed-5959/celebv_hq_head_talking/resolve/main/{clip}", timeout=120).content)
            return True

        yield stem, stem.rsplit("_", 1)[0], fetch


def raw_targets(rng, n):
    """Random 4 s segments of raw CelebV-HQ YouTube videos (films, trailers, vlogs...).
    Uses zips 0001+ so they do not overlap the celebv_raw real set (zip 0000)."""
    from remotezip import RemoteZip

    zips = [f"videos_{i:04d}.zip" for i in range(1, 6)]
    per_zip = -(-n // len(zips))
    for zname in zips:
        z = RemoteZip(f"{HF}/mengtingwei/celebv-HQ-raw-videos/resolve/main/{zname}")
        members = [i.filename for i in z.infolist() if i.filename.endswith(".mp4")]
        rng.shuffle(members)
        for m in members[:per_zip]:
            yt = Path(m).stem

            def fetch(dst, m=m, z=z):
                full = dst.with_name("full.mp4")
                full.write_bytes(z.read(m))
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                            "-of", "csv=p=0", str(full)], capture_output=True, text=True).stdout or 0)
                if dur < 6:
                    return False
                start = rng.uniform(1, dur - 5)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(full), "-t", "4",
                                "-vf", "scale=-2:'min(1080,ih)'", "-c:v", "libx264", "-crf", "12", "-an", str(dst)],
                               check=True)
                full.unlink()
                return True

            yield yt, yt, fetch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=150)
    ap.add_argument("--swaps-per-clip", type=int, default=2)
    ap.add_argument("--targets", choices=["clips", "raw"], default="clips",
                    help="clips: short CelebV talking heads; raw: segments of raw CelebV YouTube videos")
    ap.add_argument("--faces", choices=["openai", "ffhq"], default="openai",
                    help="identity photos swapped in: OpenAI-generated or real FFHQ faces")
    ap.add_argument("--tag", default="ffusion", help="source-name prefix and manifest name")
    ap.add_argument("--keep", action="store_true",
                    help="also keep the real and swapped videos in data/videos/<tag>/ (+ data/videos/<tag>.csv)")
    args = ap.parse_args()
    rng = random.Random(1)

    faces = sorted((DATA / "openai" / "orig").glob("*.png")) if args.faces == "openai" \
        else sorted((DATA / "img256" / "ffhq").glob("*.jpg"))
    man = DATA / "video_manifest" / ("facefusion.csv" if args.tag == "ffusion" else f"{args.tag}.csv")
    man.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if man.exists():
        done = {r["video"].split("/", 1)[1].split("__")[0] for r in csv.DictReader(open(man))}
    else:
        with open(man, "w", newline="") as f:
            csv.writer(f).writerow(["path", "source", "kind", "label", "video", "split"])

    targets = clip_targets(rng, args.clips) if args.targets == "clips" else raw_targets(rng, args.clips)
    k = 0
    for i, (stem, group, fetch) in enumerate(targets):
        k += args.swaps_per_clip
        if stem in done:
            continue
        split = split_of(f"celebv/{group}")
        quality = rng.choice([70, 80, 90, 95])
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw.mp4"
            try:
                if not fetch(raw):
                    continue
            except Exception as e:
                print("  fetch failed", stem, e, flush=True)
                continue
            real = tmp / "real.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264",
                            "-crf", str(ff_quality_to_crf(quality)), "-an", str(real)], check=True)
            if not frames_to_faces(real, f"{args.tag}_real", stem, "real", 0, split, rows):
                print(f"{i + 1}/{args.clips} {stem}: no face, skipped", flush=True)
                continue
            kept = []
            if args.keep:
                kept.append(keep(real, args.tag, "real", stem, 0, group, split))
            for j in range(args.swaps_per_clip):
                model = MODELS[(k + j) % len(MODELS)]
                boost = rng.choice(BOOSTS)
                out = tmp / f"swap{j}.mp4"
                if swap(rng.choice(faces), raw, out, model, boost, quality):
                    frames_to_faces(out, f"{args.tag}_{model}", f"{stem}__{boost}", "faceswap", 1, split, rows)
                    if args.keep:
                        kept.append(keep(out, args.tag, model, f"{stem}__{boost}", 1, group, split))
                else:
                    print("  swap failed", stem, model, flush=True)
        with open(man, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        if kept:
            vman = DATA / "videos" / f"{args.tag}.csv"
            new = not vman.exists()
            with open(vman, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["path", "source", "label", "group", "split"])
                w.writerows(kept)
        print(f"{i + 1}/{args.clips} {stem} ({split}) frames {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
