"""More AI-generated videos with people: Rapidata text-to-video preference sets.

  uv run fetch_rapidata.py

Repos: Rapidata/text-2-video-human-preferences-{veo3, veo3.1, veo2}. Each holds videos
from several generators (Veo 2/3/3.1, Sora, Ray2, Pika, Hunyuan, Wan, ...), named
<prompt>_<model>_<seed>.mp4. Only videos with a detectable face are kept.
  loose face frames -> data/img256/video/rp_<model>/, manifest data/video_manifest/rapidata.csv
  video files       -> data/videos/rapidata/<model>/, manifest data/videos/rapidata.csv
Split by prompt (all models' videos of one prompt share a split).
"""
import csv
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import HfApi

from fetch_video import DATA, HF, frames_to_faces

REPOS = ["Rapidata/text-2-video-human-preferences-veo3", "Rapidata/text-2-video-human-preferences-veo3.1",
         "Rapidata/text-2-video-human-preferences-veo2"]


def model_of(repo, name):
    if repo.endswith("veo3.1"):
        return "veo3.1"
    return name.split("_")[1]


def main():
    jobs = []
    for repo in REPOS:
        tag = repo.rsplit("-", 1)[1]
        for f in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True):
            if f.path.endswith(".mp4"):
                name = Path(f.path).stem
                jobs.append((repo, f.path, model_of(repo, name), f"{tag}/{name.split('_')[0]}"))
    print("videos", len(jobs), flush=True)

    def get(job):
        repo, path, _, _ = job
        for _ in range(3):
            try:
                r = requests.get(f"{HF}/{repo}/resolve/main/{path}", timeout=180)
                r.raise_for_status()
                return job, r.content
            except Exception:
                pass
        return job, None

    frame_rows, video_rows = [], []
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(8) as ex:
        for i, (job, raw) in enumerate(ex.map(get, jobs)):
            if raw is None:
                continue
            repo, path, model, prompt = job
            split = "test" if random.Random(prompt).random() < 0.2 else "train"
            vid = f"{repo.rsplit('-', 1)[1]}_{Path(path).stem}"
            p = Path(tmp) / "v.mp4"
            p.write_bytes(raw)
            n = frames_to_faces(p, f"rp_{model}", vid, "t2v", 1, split, frame_rows)
            if n:
                dst = DATA / "videos" / "rapidata" / model / f"{vid}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
                video_rows.append([str(dst.relative_to(DATA)), f"rp_{model}", 1, f"rp/{prompt}", split])
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} videos, {len(video_rows)} with faces", flush=True)

    for path, header, rows in [(DATA / "video_manifest" / "rapidata.csv", ["path", "source", "kind", "label", "video", "split"], frame_rows),
                               (DATA / "videos" / "rapidata.csv", ["path", "source", "label", "group", "split"], video_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print("done", len(video_rows), "videos with faces,", len(frame_rows), "frames")


if __name__ == "__main__":
    main()
