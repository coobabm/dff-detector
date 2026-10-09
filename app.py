"""Web UI for AI-face detection.

  uv run app.py   ->  http://127.0.0.1:7860

"一键检测" tab: one upload box for images and videos, routed to specialist models:
  image -> all_openai (AI images) + Grad-CAM
  video -> all_video (AI-generated video, loose face crops)
           + face-swap detector runs_swap/b4 (tight 380 px face crops), if trained
  Fake if any detector is above its threshold; the breakdown says which kind.
"单图·选模型" tab: pick any trained image model by hand.
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

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
IMAGE_MODEL, VIDEO_MODEL = "all_openai", "all_video"
SWAP_CKPT = ROOT / "runs_swap" / "b4" / "model.pt"
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


@lru_cache(maxsize=1)
def load_swap():
    """Face-swap detector (EfficientNet-B4) + its validation threshold, or None if not trained yet."""
    if not SWAP_CKPT.exists():
        return None
    from train_swap import build as build_swap

    ck = torch.load(SWAP_CKPT, map_location="cpu")
    m = build_swap(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(DEVICE).eval(), ck["threshold"]


@torch.no_grad()
def analyze_video(path):
    import cv2

    from extract_tight import tight_crop, video_frames
    from video_scan import faces_in

    frames = video_frames(path, 16)
    rows, gallery = [], []

    # 1) AI-generated video: loose crops, same framing as the image models.
    if VIDEO_MODEL in run_names():
        m, ps = load(VIDEO_MODEL), []
        for _, f in frames:
            faces = faces_in(f)
            if faces:
                crop = max(faces, key=lambda r: r[0][2] * r[0][3])[1]
                p = torch.sigmoid(m(tf(crop).unsqueeze(0).to(DEVICE))).item()
                ps.append(p)
                if len(ps) % 2:
                    gallery.append((crop, f"AI视频分 {p:.2f}"))
        if ps:
            rows.append(("AI 生成视频（Veo / Kling / Sora 等）", float(np.median(ps)), 0.5, len(ps)))

    # 2) Face swap: tight crops, EfficientNet-B4.
    swap = load_swap()
    if swap:
        from train_swap import eval_tf as swap_tf

        m, thr = swap
        ps = []
        for _, f in frames:
            c = tight_crop(f)
            if c is None:
                continue
            im = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            p = torch.sigmoid(m(swap_tf(im).unsqueeze(0).to(DEVICE))).item()
            ps.append(p)
            if len(ps) % 2:
                gallery.append((im, f"换脸分 {p:.2f}"))
        if ps:
            rows.append(("换脸（FaceFusion / DeepFaceLab 等）", float(np.median(ps)), thr, len(ps)))

    if not rows:
        return None, None, "### 没有检测到人脸\n\n视频里需要有清晰可见的人脸。", []
    flagged = [r for r in rows if r[1] > r[2]]
    if flagged:
        kind = max(flagged, key=lambda r: r[1] - r[2])[0]
        verdict = f"### 判定：疑似 AI 伪造\n\n最像：**{kind}**"
    else:
        verdict = "### 判定：未发现伪造痕迹"
    table = "| 检测器 | 中位数分数 | 阈值 | 用到的帧 | 结果 |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {name} | {p:.2f} | {thr:.2f} | {n} | {'⚠️ 可疑' if p > thr else '✅ 正常'} |" for name, p, thr, n in rows)
    if not swap:
        table += "\n\n换脸检测器还没训练好，本次只用了 AI 生成视频检测器。"
    top = max(rows, key=lambda r: r[1] - r[2])
    md = verdict + "\n\n" + table + "\n\n分数为 16 帧的中位数，超过阈值判为可疑。结果仅供参考。"
    return {"可疑": top[1], "正常": 1 - top[1]}, None, md, gallery


def analyze_any(file):
    if file is None:
        return None, None, "", []
    path = Path(file)
    if path.suffix.lower() in VIDEO_EXT:
        return analyze_video(str(path))
    try:
        img = Image.open(path)
    except Exception:
        return None, None, "### 无法读取这个文件\n\n请上传图片（JPG / PNG / HEIC）或视频（MP4 / MOV）。", []
    run = IMAGE_MODEL if IMAGE_MODEL in run_names() else (run_names() or [None])[0]
    lbl, cam, md = analyze(img, run)
    return lbl, cam, md, []


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
    gr.Markdown("# AI 生成人脸鉴别\n上传含**人脸**的图片或视频，自动选择合适的检测模型。")
    with gr.Tab("一键检测"):
        with gr.Row():
            with gr.Column():
                up = gr.File(label="上传图片或视频（JPG / PNG / HEIC / MP4 / MOV）", type="filepath",
                             file_types=["image", "video", ".heic", ".heif"])
                go = gr.Button("检测", variant="primary")
            with gr.Column():
                umd = gr.Markdown()
                ulbl = gr.Label(label="概率")
                ucam = gr.Image(label="图片：模型关注区域（越红越像 AI）", height=320)
                ugal = gr.Gallery(label="视频：检测到的人脸帧", columns=4, height=320)
        go.click(analyze_any, up, [ulbl, ucam, umd, ugal])
        up.upload(analyze_any, up, [ulbl, ucam, umd, ugal])
    with gr.Tab("单图·选模型"):
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
        value = IMAGE_MODEL if IMAGE_MODEL in names else (names[0] if names else None)
        return gr.update(choices=names, value=value), results_table()

    demo.load(refresh, None, [run, table])

if __name__ == "__main__":
    demo.launch()
