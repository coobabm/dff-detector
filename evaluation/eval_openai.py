"""Test every trained model on OpenAI-generated faces vs real DFF test faces.

  uv run evaluation/eval_openai.py
"""
import sys  # noqa: E402
from pathlib import Path as _P

sys.path[:0] = [str(_P(__file__).resolve().parents[1] / d) for d in ("", "detector", "training", "data_prep", "evaluation")]

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from train import DATA, build_model, predict

ROOT = Path(__file__).resolve().parents[1]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = pd.read_csv(DATA / "manifest.csv")
    real = m[(m.split == "test") & (m.source == "wiki")]
    fake = pd.DataFrame({"path": sorted(str(p.relative_to(DATA)) for p in (DATA / "img256" / "openai").glob("*.jpg"))})
    fake["label"] = 1
    print(f"real {len(real)}  openai fakes {len(fake)}")
    for run in sorted(p for p in (ROOT / "runs").iterdir() if (p / "model.pt").exists()):
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        model.to(device)
        p_real, p_fake = predict(model, real, device), predict(model, fake, device)
        res = {
            "auc": float(roc_auc_score([0] * len(p_real) + [1] * len(p_fake), list(p_real) + list(p_fake))),
            "fake_recall": float((p_fake > 0.5).mean()),
            "real_acc": float((p_real < 0.5).mean()),
            "n": len(p_fake),
        }
        json.dump(res, open(run / "openai.json", "w"), indent=1)
        print(run.name, {k: round(v, 3) for k, v in res.items()})


if __name__ == "__main__":
    main()
