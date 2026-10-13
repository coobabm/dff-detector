"""Turn the image ensemble's score into an "AI rate" (calibrated probability).

  uv run evaluation/calibrate.py   # -> runs/fm_ens/calibration.json

Uses the cached held-out scores (results/scores/, fm_compare.py). Prior is 50/50: real and fake
each carry half the weight, and inside a class every test set weighs the same, so one big set
(e.g. FairFace) does not decide the curve. Each set is split in half: fit on one half, check on the
other (expected calibration error, ECE). Isotonic regression and Platt scaling are both fitted; the one
with the lower held-out ECE is saved as a lookup table. The decision threshold is not changed.

AI rate depends on the prior: with share pi of AI images among uploads instead of 50%,
  rate_pi = r * pi / (r * pi + (1 - r) * (1 - pi))   (r = the 50/50 rate stored here).
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import fm_detector as F
from fm_compare import model_scores

ROOT = Path(__file__).resolve().parents[1]
REAL = ["photo_real", "video_real_frames", "fairface", "openfake_real", "reddit_real", "gan_set_real"]
FAKE = ["dff_fake", "openai", "stylegan", "ai_video_frames", "swap_frames", "sfhq", "reddit_ai", "openfake",
        "rapidata_unseen"]


def weighted(ens, sets, label, rng):
    """(scores, labels, weights) for fit and check halves; each set gets total weight 0.5 / len(sets)."""
    halves = [[], []]
    for s in sets:
        x = ens[s][rng.permutation(len(ens[s]))]
        for h, part in enumerate(np.array_split(x, 2)):
            halves[h].append((part, np.full(len(part), label), np.full(len(part), 0.5 / len(sets) / len(part))))
    return [tuple(np.concatenate(c) for c in zip(*h)) for h in halves]


def ece(p, y, w, bins=10):
    """Weighted expected calibration error: |mean predicted - observed AI share| per score bin."""
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if w[m].sum() > 0:
            err += w[m].sum() * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return err / w.sum()


def main():
    members = json.loads((ROOT / "runs" / "fm_ens" / "ensemble.json").read_text())["members"]
    _, tests = F.data()
    cache = {m: model_scores(m, tests) for m in members}
    ens = {k: np.mean([cache[m][k] for m in members], 0) for k in REAL + FAKE}
    rng = np.random.default_rng(0)
    (rf, rl, rw), (rc, rcl, rcw) = weighted(ens, REAL, 0, rng)
    (ff, fl, fw), (fc, fcl, fcw) = weighted(ens, FAKE, 1, rng)
    x, y, w = np.r_[rf, ff], np.r_[rl, fl], np.r_[rw, fw]
    xc, yc, wc = np.r_[rc, fc], np.r_[rcl, fcl], np.r_[rcw, fcw]

    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(x, y, sample_weight=w)
    logit = lambda s: np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6).fit(logit(x)[:, None], y, sample_weight=w)
    raw_ece = ece(xc, yc, wc)
    iso_ece = ece(iso.predict(xc), yc, wc)
    platt_ece = ece(platt.predict_proba(logit(xc)[:, None])[:, 1], yc, wc)
    print(f"held-out ECE  raw score {raw_ece:.3f} | isotonic {iso_ece:.3f} | Platt {platt_ece:.3f}")

    grid = np.r_[np.linspace(0, 0.9, 91), np.linspace(0.9, 1, 101)[1:]]
    if iso_ece <= platt_ece:
        method, table = "isotonic", iso.predict(grid)
    else:
        method, table = "platt", platt.predict_proba(logit(grid)[:, None])[:, 1]
    table = np.maximum.accumulate(table)  # monotone lookup for np.interp
    for s in (0.05, 0.1, 0.3, 0.5, 0.715, 0.9, 0.99):
        print(f"score {s:<5} -> AI rate {np.interp(s, grid, table):.1%}")
    out = {"method": method, "prior": 0.5, "x": [round(v, 4) for v in grid], "y": [round(float(v), 4) for v in table],
           "ece_heldout": {"raw": round(raw_ece, 4), "isotonic": round(iso_ece, 4), "platt": round(platt_ece, 4)},
           "members": members,
           "note": "AI rate under a 50/50 prior of real and AI images; see calibrate.py for other priors."}
    (ROOT / "runs" / "fm_ens" / "calibration.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
