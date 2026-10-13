"""Keep a slice of public face-swap / real videos for the video face-swap detector.

  uv run data_prep/fetch_public_videos.py

FF++ C23: 200 real + 40 per manipulation method. DF40 (built on FF++ identities):
60 each of inswap / simswap / uniface. Split by FF++ target id: >= 800 is test.
Videos -> data/videos/public/<source>/<name>.mp4, manifest data/videos/public.csv.
"""
import csv
import random
import time
from pathlib import Path

from remotezip import RemoteZip

DATA = Path(__file__).resolve().parents[1] / "data"
HF = "https://huggingface.co/datasets"
FFPP = "FaceForensics++_C23"
JOBS = [
    # source, repo, zip, member filter, how many, label
    ("ffpp_real", "bitmind/FaceForensicsC23", f"{FFPP}.zip", f"{FFPP}/real/", 200, 0),
    *[(f"ffpp_{m.lower()}", "bitmind/FaceForensicsC23", f"{FFPP}.zip", f"{FFPP}/fake/{m}/", 40, 1)
      for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]],
    ("df40_inswap", "ohjoonhee/DF40-Fake-Videos", "train/inswap.zip", "inswap/", 60, 1),
    ("df40_simswap", "ohjoonhee/DF40-Fake-Videos", "train/simswap.zip", "simswap/", 60, 1),
    ("df40_uniface", "ohjoonhee/DF40-Fake-Videos", "test/uniface.zip", "uniface/", 60, 1),
]


def main():
    rng = random.Random(0)
    man = DATA / "videos" / "public.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, repo, zname, prefix, n, label in JOBS:
        with RemoteZip(f"{HF}/{repo}/resolve/main/{zname}") as z:
            names = sorted(i.filename for i in z.infolist() if i.filename.startswith(prefix) and i.filename.endswith(".mp4"))
            rng.shuffle(names)
            got = 0
            for name in names[:n]:
                stem = Path(name).stem
                head = stem.split("_")[0]
                if not head.isdigit():  # e.g. DF40 celeb-df style "id25_..." names: no FF++ id, skip
                    continue
                ident = int(head)
                dst = DATA / "videos" / "public" / source / f"{stem}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    for attempt in range(4):
                        try:
                            dst.write_bytes(z.read(name))
                            break
                        except Exception as e:
                            print("  retry", name, e, flush=True)
                            time.sleep(5 * (attempt + 1))
                    else:
                        continue
                rows.append([str(dst.relative_to(DATA)), source, label, f"ffpp/{ident}",
                             "test" if ident >= 800 else "train"])
                got += 1
        print(source, got, flush=True)
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "group", "split"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
