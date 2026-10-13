"""Cross-generator table for the web UI: every image model x every held-out set.

  uv run arms_race.py   # -> results/arms_race.json (read by app.py, "arms race" tab)

Rows (sets) are grouped into real sets (share judged real) and fake sets (share caught). DFF fakes are
split back into SD text2img / SD inpainting / InsightFace swap, so the original arms-race question
(train on one generator, test on the others) stays visible. Every model is put at the same operating
point: threshold = 97th percentile of its scores on half A of the held-out real photos; numbers are on
half B and on all other sets. Published detectors (UnivFD / NPR / B-Free) are slow, so they run on a
capped subset of each set (n is stored). Scores of our models come from results/scores/ (fm_compare.py).
"""
import json
from pathlib import Path

import numpy as np

import app
import external_detectors as X
import fm_detector as F
from fm_compare import model_scores
from PIL import Image
from train import DATA

ROOT = Path(__file__).parent
OURS = ["t2i", "inpaint", "insight", "all", "all_image", "all_shot", "fm_ft_dinov2_b", "fm_ft_clip_l14"]
ENSEMBLE = ["fm_ft_clip_l14_lora", "fm_ft_dinov2_b"]  # = runs/fm_ens (live)
EXT_CAP = 150
REAL = ["photo_real", "video_real_frames", "fairface", "openfake_real", "reddit_real"]
FAKE = ["dff_text2img", "dff_inpainting", "dff_insight", "openai", "stylegan", "ai_video_frames", "swap_frames",
        "sfhq", "reddit_ai", "openfake", "rapidata_unseen"]


def split_dff(tests, sc):
    """dff_fake -> one set per DFF generator (scores stay aligned with the rows)."""
    src = tests["dff_fake"].source.values
    out = {k: v for k, v in sc.items() if k != "dff_fake"}
    for g in ("text2img", "inpainting", "insight"):
        out[f"dff_{g}"] = sc["dff_fake"][src == g]
    return out


def rates(sc, half_a, half_b):
    thr = float(np.quantile(sc["photo_real"][half_a], .97))
    r = {"threshold": thr}
    for k in REAL:
        s = sc[k][half_b] if k == "photo_real" else sc[k]
        r[k] = round(float(np.mean(s <= thr)), 3)
    for k in FAKE:
        r[k] = round(float(np.mean(sc[k] > thr)), 3)
    return r


def main():
    app.DEVICE = F.DEVICE
    _, tests = F.data()
    halves = F.halves_of(tests)
    n = {k: int(len(d)) for k, d in tests.items()}
    rows = {}
    cache = {m: split_dff(tests, model_scores(m, tests)) for m in sorted(set(OURS + ENSEMBLE))}
    for m in OURS:
        rows[m] = rates(cache[m], *halves)
        print(m, rows[m], flush=True)
    ens = {k: np.mean([cache[m][k] for m in ENSEMBLE], 0) for k in cache[ENSEMBLE[0]]}
    rows["fm_ens"] = rates(ens, *halves)
    print("fm_ens", rows["fm_ens"], flush=True)

    # Published detectors: capped subsets, same operating point (97% of the subset's real photos, half A).
    sub = {k: (d if len(d) <= EXT_CAP else d.sample(EXT_CAP, random_state=1)) for k, d in tests.items()}
    sub["photo_real"] = tests["photo_real"].sample(2 * EXT_CAP, random_state=1)
    rng = np.random.default_rng(0)
    idx = rng.permutation(2 * EXT_CAP)
    ext_halves = (idx[:EXT_CAP], idx[EXT_CAP:])
    for det in X.available():
        sc = {k: np.array([X.score(det, Image.open(DATA / p).convert("RGB"), F.DEVICE) for p in d.path])
              for k, d in sub.items()}
        rows[f"ext_{det}"] = rates(split_dff(sub, sc), *ext_halves)
        print(det, rows[f"ext_{det}"], flush=True)
    n_ext = {k: int(len(d)) for k, d in split_dff(sub, {k: np.zeros(len(d)) for k, d in sub.items()}).items()}
    out = {"real_sets": REAL, "fake_sets": FAKE, "models": rows,
           "n": {k: int(len(v)) for k, v in split_dff(tests, {k: np.zeros(len(d)) for k, d in tests.items()}).items()},
           "n_external": n_ext, "ensemble_members": ENSEMBLE}
    json.dump(out, open(ROOT / "results" / "arms_race.json", "w"), indent=1)


if __name__ == "__main__":
    main()
