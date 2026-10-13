"""Score every image model on every held-out set once, then compare single models and averages.

  uv run evaluation/fm_compare.py all_shot fm_ft_dinov2_b fm_ft_clip_l14

Scores are cached in results/scores/<model>.npz. Every row uses its own threshold: the 97th percentile
of its scores on half A of the held-out real photos; numbers are on half B and on every other set.
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

import app
import fm_detector as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "scores"


def fingerprint(df):
    import hashlib

    return hashlib.md5("\n".join(df.path).encode()).hexdigest()


def model_scores(name, tests, flip=False):
    """Cached scores; reused only when every test set has exactly the same rows (path fingerprint).
    flip=True: scores of the mirrored images (test-time augmentation = average with the plain scores)."""
    f = CACHE / f"{name}{'__flip' if flip else ''}.npz"
    if f.exists():
        d = dict(np.load(f))
        if all(k in d and f"__fp_{k}" in d and str(d[f"__fp_{k}"]) == fingerprint(tests[k]) for k in tests):
            return {k: d[k] for k in tests}
    m = app.load(name)
    from torchvision import transforms as T

    tf = T.Compose([T.RandomHorizontalFlip(p=1.0), app.image_tf(name)]) if flip else app.image_tf(name)
    d = {k: F.scores(m, df, tf) for k, df in tests.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d, **{f"__fp_{k}": np.array(fingerprint(df)) for k, df in tests.items()})
    return d


def main():
    names = sys.argv[1:]
    app.DEVICE = F.DEVICE
    _, tests = F.data()
    halves = F.halves_of(tests)
    sc = {n: model_scores(n, tests) for n in names}
    rows = {n: F.summarize(n, sc[n], tests, *halves) for n in names}
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            avg = {t: np.mean([sc[n][t] for n in combo], 0) for t in tests}
            rows["+".join(combo)] = F.summarize("avg(" + "+".join(combo) + ")", avg, tests, *halves)
    json.dump(rows, open(ROOT / "results" / "fm_compare.json", "w"), indent=1)
    sets = [k for k in rows[names[0]] if k != "threshold"]
    print("\n" + "set".ljust(20) + "".join(n[:14].rjust(16) for n in rows))
    for s in sets:
        print(s.ljust(20) + "".join(f"{rows[n][s]:16.3f}" for n in rows))


if __name__ == "__main__":
    with torch.no_grad():
        main()
