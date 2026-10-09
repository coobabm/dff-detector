"""Web UI: upload a face image, get AI-generated probability + Grad-CAM heatmap.

  uv run app.py   ->  http://127.0.0.1:7860
"""
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # iPhone HEIC/HEIF uploads
from torchvision import transforms as T

from train import FAKES, MEAN, STD, build_model

RUNS = Path(__file__).parent / "runs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LABELS = {"text2img": "SD 文生图", "inpainting": "SD Inpainting", "insight": "InsightFace 换脸"}
tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def run_names():
    return sorted(p.name for p in RUNS.glob("*") if (p / "model.pt").exists())


@lru_cache(maxsize=8)
def load(name):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256), Image.BICUBIC)


def analyze(img, run):
    if img is None or not run:
        return None, None, ""
    model = load(run)
    im = square(ImageOps.exif_transpose(img).convert("RGB"))
    x = tf(im).unsqueeze(0).to(DEVICE)
    feats = model.forward_features(x)
    feats.retain_grad()
    logit = model.forward_head(feats).squeeze()
    p = torch.sigmoid(logit).item()
    logit.backward()
    # Grad-CAM toward "fake".
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=im.size[::-1], mode="bilinear")[0, 0].detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    overlay = Image.blend(im, Image.fromarray(heat), 0.45)
    verdict = "AI 生成 / 篡改" if p > 0.5 else "真实照片"
    md = f"### 判定：{verdict}\n\nAI 生成概率 **{p:.1%}**（模型：`{run}`）"
    return {"AI 生成": p, "真实": 1 - p}, overlay, md


def results_table():
    rows = []
    for name in run_names():
        f = RUNS / name / "results.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        trained = " + ".join(LABELS[s] for s in r["fakes"])
        cells = []
        for s in FAKES:
            mark = "（训练过）" if s in r["fakes"] else "（没见过）"
            cells.append(f"{r['test'][s]['auc']:.3f} / {r['test'][s]['acc']:.1%} {mark}")
        rows.append(f"| `{name}` | {trained} | {r['test']['real_acc']:.1%} | " + " | ".join(cells) + " |")
    if not rows:
        return "还没有训练结果。"
    head = "| 模型 | 训练用的假图 | 真图准确率 | " + " | ".join(LABELS[s] + " AUC / Acc" for s in FAKES) + " |\n"
    head += "|---" * (3 + len(FAKES)) + "|\n"
    return head + "\n".join(rows) + "\n\n测试集为训练时完全未见的人物（IMDB-WIKI 子文件夹 90–99）。"


with gr.Blocks(title="AI 人脸鉴别") as demo:
    gr.Markdown("# AI 生成人脸鉴别\n基于 DeepFakeFace 训练的 EfficientNet-B0。上传一张**人脸**图片（最好是正脸裁切）。")
    with gr.Tab("检测"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="上传图片", height=360)
                run = gr.Dropdown(run_names(), label="模型", allow_custom_value=True)
                btn = gr.Button("检测", variant="primary")
            with gr.Column():
                md = gr.Markdown()
                lbl = gr.Label(label="概率")
                cam = gr.Image(label="Grad-CAM：模型关注区域（越红越像 AI）", height=360)
        btn.click(analyze, [inp, run], [lbl, cam, md])
        inp.change(analyze, [inp, run], [lbl, cam, md])
    with gr.Tab("跨生成器测试（arms race）"):
        gr.Markdown("用一种生成方式训练，拿另外两种测试：没见过的生成器上分数掉多少，就是检测器的泛化差距。")
        table = gr.Markdown()

    def refresh():
        names = run_names()
        value = "all" if "all" in names else (names[0] if names else None)
        return gr.update(choices=names, value=value), results_table()

    demo.load(refresh, None, [run, table])

if __name__ == "__main__":
    demo.launch()
