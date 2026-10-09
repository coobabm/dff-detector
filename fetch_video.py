"""Download a small slice of AI / real face videos and turn them into face-frame images.

  uv run fetch_video.py            # all sources
  uv run fetch_video.py ffpp thb   # some sources

Only the chosen videos are fetched (remote zip members via HTTP range requests),
so the full multi-GB archives are never downloaded. For each video, FRAMES evenly
spaced frames are taken, faces found with YuNet (same crop as video_scan.py) and
saved as 256x256 JPEGs:

  data/img256/video/<source>/<video_id>_<k>.jpg
  data/video_manifest/<source>.csv   path, source, kind, label, video, split

Split is by video (and for FF++ by source identity), never by frame.
"""
import csv
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import requests
from huggingface_hub import HfApi
from remotezip import RemoteZip

from video_scan import faces_in

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "img256" / "video"
HF = "https://huggingface.co/datasets"
FRAMES = 8
rng = random.Random(0)

# source -> (repo, zip member path, videos to take, kind, label)
ZIP_SOURCES = {
    "df40_inswap": ("ohjoonhee/DF40-Fake-Videos", "train/inswap.zip", 120, "faceswap", 1),
    "df40_simswap": ("ohjoonhee/DF40-Fake-Videos", "train/simswap.zip", 120, "faceswap", 1),
    "df40_uniface": ("ohjoonhee/DF40-Fake-Videos", "test/uniface.zip", 120, "faceswap", 1),
    "df40_heygen": ("ohjoonhee/DF40-Fake-Videos", "test/heygen.zip", 120, "talkinghead", 1),
    "veo3": ("34data/gen-videos-veo3", "data_001.zip", 314, "t2v", 1),
    "sora2": ("34data/gen-videos-sora2", "data_001.zip", 302, "t2v", 1),
    "kling": ("34data/gen-videos-kling", "data_001.zip", 300, "t2v", 1),
    "wan2": ("34data/gen-videos-wan2", "data_001.zip", 300, "t2v", 1),
    "sora1": ("jian-0/GenVidBench", "GenVidBench/6.7m/OpenAI_Sora.zip", 51, "t2v", 1),
    "keling": ("jian-0/GenVidBench", "GenVidBench/6.7m/keling.zip", 300, "t2v", 1),
}
FFPP = "FaceForensics++_C23"
FFPP_FAKES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def frames_to_faces(video_path, source, vid, kind, label, split, rows):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        return 0
    got = 0
    for k, idx in enumerate(sorted({int(n * (i + 0.5) / FRAMES) for i in range(FRAMES)})):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = faces_in(frame)
        if not faces:
            continue
        # Largest face only: one sample per frame.
        box, crop = max(faces, key=lambda f: f[0][2] * f[0][3])
        dst = OUT / source / f"{vid}_{k}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, quality=95)
        rows.append([str(dst.relative_to(DATA)), source, kind, label, f"{source}/{vid}", split])
        got += 1
    return got


def split_of(key, test_frac=0.2):
    return "test" if random.Random(key).random() < test_frac else "train"


def from_zip(source, repo, member, take, kind, label, rows, names=None, split_fn=None):
    url = f"{HF}/{repo}/resolve/main/{member}"
    with RemoteZip(url) as z:
        vids = [i.filename for i in z.infolist()
                if i.filename.lower().endswith((".mp4", ".mov")) and "__MACOSX" not in i.filename]
        if names is not None:
            vids = [v for v in vids if v in names]
        else:
            rng.shuffle(vids)
            vids = vids[:take]
        faces = 0
        with tempfile.TemporaryDirectory() as tmp:
            for i, v in enumerate(vids):
                vid = Path(v).stem.replace(" ", "_")
                split = split_fn(v) if split_fn else split_of(f"{source}/{vid}")
                p = Path(tmp) / f"v{Path(v).suffix}"
                for attempt in range(4):
                    try:
                        p.write_bytes(z.read(v))
                        break
                    except Exception as e:
                        print(f"  retry {v}: {e}", flush=True)
                        time.sleep(5 * (attempt + 1))
                else:
                    continue
                faces += frames_to_faces(p, source, vid, kind, label, split, rows)
                if i % 25 == 0:
                    print(f"  {source} {i}/{len(vids)} videos, {faces} faces", flush=True)
    print(f"{source}: {len(vids)} videos -> {faces} face frames", flush=True)


def ffpp(rows):
    """FF++ C23: 300 real + 60 per fake method. Split by FF++ id: ids >= 800 are test."""
    test = lambda v: "test" if int(Path(v).stem.split("_")[0]) >= 800 else "train"
    url = f"{HF}/bitmind/FaceForensicsC23/resolve/main/{FFPP}.zip"
    with RemoteZip(url) as z:
        names = [i.filename for i in z.infolist() if i.filename.endswith(".mp4")]
    real = sorted(n for n in names if "/real/" in n)
    rng.shuffle(real)
    from_zip("ffpp_real", "bitmind/FaceForensicsC23", f"{FFPP}.zip", 0, "real", 0, rows,
             names=set(real[:300]), split_fn=test)
    for m in FFPP_FAKES:
        fakes = sorted(n for n in names if f"/fake/{m}/" in n)
        rng.shuffle(fakes)
        from_zip(f"ffpp_{m.lower()}", "bitmind/FaceForensicsC23", f"{FFPP}.zip", 0, "faceswap", 1, rows,
                 names=set(fakes[:60]), split_fn=test)


def thb(rows):
    """TalkingHeadBench: up to 60 train + 20 test videos per generator."""
    files = [f.path for f in HfApi().list_repo_tree("luchaoqi/TalkingHeadBench", repo_type="dataset",
                                                     path_in_repo="fake", recursive=True)
             if getattr(f, "size", None) and f.path.endswith(".mp4")]
    by_gen = {}
    for f in files:
        parts = f.split("/")
        if len(parts) == 4 and parts[2] in ("train", "test"):
            by_gen.setdefault((parts[1], parts[2]), []).append(f)
    for (gen, split), fs in sorted(by_gen.items()):
        rng.shuffle(fs)
        source, faces = f"thb_{gen.lower()}", 0
        with tempfile.TemporaryDirectory() as tmp:
            for f in fs[: 60 if split == "train" else 20]:
                p = Path(tmp) / "v.mp4"
                p.write_bytes(requests.get(f"{HF}/luchaoqi/TalkingHeadBench/resolve/main/{f}", timeout=120).content)
                faces += frames_to_faces(p, source, Path(f).stem, "talkinghead", 1, split, rows)
        print(f"{source} {split}: {faces} face frames", flush=True)


def celebv_clips(rows, take=600):
    """Real CelebV-HQ talking-head clips (YouTube). Split by YouTube id."""
    files = sorted(f.path for f in HfApi().list_repo_tree("saeed-5959/celebv_hq_head_talking", repo_type="dataset",
                                                         recursive=True) if f.path.endswith(".mp4"))
    rng.shuffle(files)
    faces = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files[:take]):
            yt = Path(f).stem.rsplit("_", 1)[0]
            p = Path(tmp) / "v.mp4"
            try:
                p.write_bytes(requests.get(f"{HF}/saeed-5959/celebv_hq_head_talking/resolve/main/{f}", timeout=120).content)
            except Exception as e:
                print("  skip", f, e, flush=True)
                continue
            faces += frames_to_faces(p, "celebv_clips", Path(f).stem, "real", 0, split_of(f"celebv/{yt}"), rows)
            if i % 50 == 0:
                print(f"  celebv_clips {i}/{take}, {faces} faces", flush=True)
    print(f"celebv_clips: {faces} face frames", flush=True)


def celebv_raw(rows, take=80):
    """Raw CelebV-HQ YouTube videos (~50 MB each), a few from the first zip."""
    from_zip("celebv_raw", "mengtingwei/celebv-HQ-raw-videos", "videos_0000.zip", take, "real", 0, rows,
             split_fn=lambda v: split_of(f"celebv/{Path(v).stem}"))


def main():
    want = sys.argv[1:] or ["ffpp", "thb", "celebv_clips", "celebv_raw", *ZIP_SOURCES]
    (DATA / "video_manifest").mkdir(parents=True, exist_ok=True)
    for s in want:
        rows = []
        if s == "ffpp":
            ffpp(rows)
        elif s == "thb":
            thb(rows)
        elif s in ("celebv_clips", "celebv_raw"):
            globals()[s](rows)
        else:
            from_zip(s, *ZIP_SOURCES[s], rows)
        # One manifest per requested source, so several processes can run in parallel.
        with open(DATA / "video_manifest" / f"{s}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "source", "kind", "label", "video", "split"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
