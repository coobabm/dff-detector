"""Keep AI-generated / talking-head videos (with faces) + extra real videos for a
tight-crop AI-video detector (same recipe as the face-swap detector).

  uv run fetch_aivid_videos.py

Fakes: 34data Veo 3 / Sora 2 / Kling / Wan 2, GenVidBench Sora / Kling,
       TalkingHeadBench (6 generators), DF40 HeyGen.
Reals: segments of raw CelebV-HQ YouTube videos (zips 0006+, unused elsewhere).
Videos without a detectable face are dropped.
Videos -> data/videos/aivid/<source>/, manifest data/videos/aivid.csv.
"""
import csv
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import requests
from huggingface_hub import HfApi
from remotezip import RemoteZip

from extract_tight import tight_crop, video_frames

DATA = Path(__file__).parent / "data"
HF = "https://huggingface.co/datasets"
OUT = DATA / "videos" / "aivid"
ZIPS = [
    ("veo3", "34data/gen-videos-veo3", "data_001.zip", 400),
    ("sora2", "34data/gen-videos-sora2", "data_001.zip", 400),
    ("kling", "34data/gen-videos-kling", "data_001.zip", 600),
    ("wan2", "34data/gen-videos-wan2", "data_001.zip", 600),
    ("sora1", "jian-0/GenVidBench", "GenVidBench/6.7m/OpenAI_Sora.zip", 60),
    ("keling", "jian-0/GenVidBench", "GenVidBench/6.7m/keling.zip", 540),
    ("df40_heygen", "ohjoonhee/DF40-Fake-Videos", "test/heygen.zip", 120),
]
rng = random.Random(0)


def has_face(path, need=3):
    return sum(tight_crop(f) is not None for _, f in video_frames(path, 6)) >= need


def split_of(key):
    return "test" if random.Random(key).random() < 0.2 else "train"


def keep(src_path, source, name, label, group, split, rows):
    dst = OUT / source / f"{name}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dst)
    rows.append([str(dst.relative_to(DATA)), source, label, group, split])


def from_zips(rows):
    for source, repo, member, take in ZIPS:
        with RemoteZip(f"{HF}/{repo}/resolve/main/{member}") as z, tempfile.TemporaryDirectory() as tmp:
            names = [i.filename for i in z.infolist()
                     if i.filename.lower().endswith((".mp4", ".mov")) and "__MACOSX" not in i.filename]
            rng.shuffle(names)
            kept = 0
            for n in names[:take]:
                p = Path(tmp) / f"v{Path(n).suffix}"
                try:
                    p.write_bytes(z.read(n))
                except Exception as e:
                    print("  skip", n, e, flush=True)
                    continue
                if has_face(p):
                    name = Path(n).stem.replace(" ", "_")
                    keep(p, source, name, 1, f"{source}/{name}", split_of(f"{source}/{name}"), rows)
                    kept += 1
        print(source, "kept", kept, flush=True)


def thb(rows, per_split={"train": 80, "test": 25}):
    files = [f.path for f in HfApi().list_repo_tree("luchaoqi/TalkingHeadBench", repo_type="dataset",
                                                     path_in_repo="fake", recursive=True)
             if f.path.endswith(".mp4") and f.path.count("/") == 3]
    by = {}
    for f in files:
        _, gen, split, _ = f.split("/")
        if split in per_split:
            by.setdefault((gen, split), []).append(f)
    with tempfile.TemporaryDirectory() as tmp:
        for (gen, split), fs in sorted(by.items()):
            rng.shuffle(fs)
            kept = 0
            for f in fs[: per_split[split]]:
                p = Path(tmp) / "v.mp4"
                p.write_bytes(requests.get(f"{HF}/luchaoqi/TalkingHeadBench/resolve/main/{f}", timeout=120).content)
                if has_face(p):
                    keep(p, f"thb_{gen.lower()}", Path(f).stem, 1, f"thb/{Path(f).stem.split('--')[0]}", split, rows)
                    kept += 1
            print(f"thb_{gen.lower()} {split} kept {kept}", flush=True)


def celebv_real(rows, take=200):
    """4 s segments of raw CelebV-HQ videos from zips not used anywhere else."""
    kept = 0
    with tempfile.TemporaryDirectory() as tmp:
        for zname in [f"videos_{i:04d}.zip" for i in range(6, 10)]:
            z = RemoteZip(f"{HF}/mengtingwei/celebv-HQ-raw-videos/resolve/main/{zname}")
            members = [i.filename for i in z.infolist() if i.filename.endswith(".mp4")]
            rng.shuffle(members)
            for m in members[: take // 4]:
                full, seg = Path(tmp) / "full.mp4", Path(tmp) / "seg.mp4"
                try:
                    full.write_bytes(z.read(m))
                except Exception as e:
                    print("  skip", m, e, flush=True)
                    continue
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                            "csv=p=0", str(full)], capture_output=True, text=True).stdout or 0)
                if dur < 6:
                    continue
                start = rng.uniform(1, dur - 5)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(full), "-t", "4",
                                "-vf", "scale=-2:'min(1080,ih)'", "-c:v", "libx264", "-crf", str(rng.choice([18, 23, 28])),
                                "-an", str(seg)], check=True)
                yt = Path(m).stem
                if has_face(seg):
                    keep(seg, "celebv_real", yt, 0, f"celebv/{yt}", split_of(f"celebv/{yt}"), rows)
                    kept += 1
    print("celebv_real kept", kept, flush=True)


def main():
    rows = []
    for step in (from_zips, thb, celebv_real):
        step(rows)
        with open(DATA / "videos" / "aivid.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "source", "label", "group", "split"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
