"""Web UI for AI-face detection (中文 / English / 한국어).

  uv run app.py   ->  http://127.0.0.1:7860

"Detect" tab: one upload box for images and videos, routed to specialist models:
  image -> all_image (AI images incl. StyleGAN; falls back to all_openai) + Grad-CAM
  video -> AI-video detector runs_aivid/b4 + face-swap detector runs_swap/b4
           (both EfficientNet-B4 on tight 380 px face crops; all_video as fallback)
  Fake if any detector is above its threshold; the breakdown says which kind.
"Pick a model" tab: run any trained image model by hand.
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
from torchvision import transforms as T

from train import FAKES, MEAN, STD, build_model

register_heif_opener()  # iPhone HEIC/HEIF uploads

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
VIDEO_MODEL = "all_video"  # fallback AI-video detector (loose crops) if the B4 one is not trained
SWAP_CKPT = ROOT / "runs_swap" / "b4" / "model.pt"
AIVID_CKPT = ROOT / "runs_aivid" / "b4" / "model.pt"


def image_model():
    """Best image model available: all_image (AI images + StyleGAN), else all_openai."""
    for name in ("all_image", "all_openai"):
        if (RUNS / name / "model.pt").exists():
            return name
    return (run_names() or [None])[0]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

LANGS = {"中文": "zh", "English": "en", "한국어": "ko"}
TXT = {
    "zh": {
        "title": "# AI 生成人脸鉴别\n上传含**人脸**的图片或视频，自动选择合适的检测模型。模型分数不是鉴定结论。",
        "tab_auto": "一键检测", "tab_pick": "单图·选模型", "tab_arms": "跨生成器测试（arms race）",
        "upload": "上传图片或视频（JPG / PNG / HEIC / MP4 / MOV）", "detect": "检测",
        "prob": "概率", "cam": "图片：模型关注区域（越红越像 AI）", "gallery": "视频：检测到的人脸帧",
        "img_in": "上传图片", "model": "模型", "cam_pick": "Grad-CAM：模型关注区域（越红越像 AI）",
        "arms_intro": "用一种生成方式训练，拿另外两种测试：没见过的生成器上分数掉多少，就是检测器的泛化差距。",
        "fake": "AI 生成", "real": "真实", "susp": "可疑", "ok": "正常",
        "v_fake_img": "AI 生成 / 篡改", "v_real_img": "真实照片",
        "img_md": "### 判定：{v}\n\nAI 生成概率 **{p:.1%}**（模型：`{m}`）",
        "det_aivid": "AI 生成视频（Veo / Kling / Sora 等）", "det_swap": "换脸（FaceFusion / DeepFaceLab 等）",
        "no_face": "### 没有检测到人脸\n\n视频里需要有清晰可见的人脸。",
        "bad_file": "### 无法读取这个文件\n\n请上传图片（JPG / PNG / HEIC）或视频（MP4 / MOV）。",
        "v_fake": "### 判定：疑似 AI 伪造\n\n最像：**{k}**", "v_real": "### 判定：未发现伪造痕迹",
        "th": "| 检测器 | 中位数分数 | 阈值 | 用到的帧 | 结果 |", "r_susp": "⚠️ 可疑", "r_ok": "✅ 正常",
        "no_swap": "换脸检测器还没训练好，本次只用了 AI 生成视频检测器。",
        "foot": "分数为 16 帧的中位数，超过阈值判为可疑。结果仅供参考。",
        "cap_aivid": "AI视频分", "cap_swap": "换脸分",
        "no_res": "还没有训练结果。", "tr_seen": "（训练过）", "tr_unseen": "（没见过）",
        "tbl_head": "| 模型 | 训练用的假图 | 真图准确率 | {cols} |",
        "tbl_foot": "测试集为训练时完全未见的人物（IMDB-WIKI 子文件夹 90–99）。",
        "gens": {"text2img": "SD 文生图", "inpainting": "SD Inpainting", "insight": "InsightFace 换脸"},
    },
    "en": {
        "title": "# AI-generated face detection\nUpload an image or video containing a **face**; the right detector is chosen automatically. Scores are not proof of authenticity.",
        "tab_auto": "Detect", "tab_pick": "Pick a model (image)", "tab_arms": "Cross-generator test (arms race)",
        "upload": "Upload an image or video (JPG / PNG / HEIC / MP4 / MOV)", "detect": "Detect",
        "prob": "Probability", "cam": "Image: where the model looks (redder = more AI-like)",
        "gallery": "Video: detected face frames",
        "img_in": "Upload image", "model": "Model", "cam_pick": "Grad-CAM: where the model looks (redder = more AI-like)",
        "arms_intro": "Train on one generator, test on the other two: the drop on unseen generators is the generalisation gap.",
        "fake": "AI-generated", "real": "Real", "susp": "Suspicious", "ok": "Normal",
        "v_fake_img": "AI-generated / manipulated", "v_real_img": "Real photo",
        "img_md": "### Verdict: {v}\n\nAI-generated probability **{p:.1%}** (model: `{m}`)",
        "det_aivid": "AI-generated video (Veo / Kling / Sora …)", "det_swap": "Face swap (FaceFusion / DeepFaceLab …)",
        "no_face": "### No face found\n\nThe video needs a clearly visible face.",
        "bad_file": "### Can't read this file\n\nPlease upload an image (JPG / PNG / HEIC) or a video (MP4 / MOV).",
        "v_fake": "### Verdict: likely AI fake\n\nMost like: **{k}**", "v_real": "### Verdict: no signs of manipulation found",
        "th": "| Detector | Median score | Threshold | Frames used | Result |", "r_susp": "⚠️ Suspicious", "r_ok": "✅ Normal",
        "no_swap": "The face-swap detector is not trained yet; only the AI-video detector was used.",
        "foot": "Scores are the median over 16 frames; above the threshold counts as suspicious. For reference only.",
        "cap_aivid": "AI-video", "cap_swap": "swap",
        "no_res": "No training results yet.", "tr_seen": " (trained on)", "tr_unseen": " (unseen)",
        "tbl_head": "| Model | Fakes trained on | Real accuracy | {cols} |",
        "tbl_foot": "Test set: people never seen in training (IMDB-WIKI folders 90–99).",
        "gens": {"text2img": "SD text2img", "inpainting": "SD inpainting", "insight": "InsightFace swap"},
    },
    "ko": {
        "title": "# AI 생성 얼굴 판별\n**얼굴**이 있는 이미지나 영상을 올리면 알맞은 탐지 모델을 자동으로 고릅니다. 모델 점수는 진위 판정의 근거가 아닙니다.",
        "tab_auto": "탐지", "tab_pick": "이미지·모델 선택", "tab_arms": "생성기 교차 테스트 (arms race)",
        "upload": "이미지 또는 영상 업로드 (JPG / PNG / HEIC / MP4 / MOV)", "detect": "탐지",
        "prob": "확률", "cam": "이미지: 모델이 주목한 영역 (빨갈수록 AI 같음)", "gallery": "영상: 검출된 얼굴 프레임",
        "img_in": "이미지 업로드", "model": "모델", "cam_pick": "Grad-CAM: 모델이 주목한 영역 (빨갈수록 AI 같음)",
        "arms_intro": "한 생성기로 학습하고 나머지 두 생성기로 테스트: 보지 못한 생성기에서 떨어지는 만큼이 일반화 격차입니다.",
        "fake": "AI 생성", "real": "진짜", "susp": "의심", "ok": "정상",
        "v_fake_img": "AI 생성 / 조작", "v_real_img": "실제 사진",
        "img_md": "### 판정: {v}\n\nAI 생성 확률 **{p:.1%}** (모델: `{m}`)",
        "det_aivid": "AI 생성 영상 (Veo / Kling / Sora 등)", "det_swap": "얼굴 교체 (FaceFusion / DeepFaceLab 등)",
        "no_face": "### 얼굴을 찾지 못했습니다\n\n영상에 얼굴이 선명하게 보여야 합니다.",
        "bad_file": "### 파일을 읽을 수 없습니다\n\n이미지(JPG / PNG / HEIC)나 영상(MP4 / MOV)을 올려 주세요.",
        "v_fake": "### 판정: AI 조작 의심\n\n가장 가까운 유형: **{k}**", "v_real": "### 판정: 조작 흔적 없음",
        "th": "| 탐지기 | 중앙값 점수 | 임계값 | 사용한 프레임 | 결과 |", "r_susp": "⚠️ 의심", "r_ok": "✅ 정상",
        "no_swap": "얼굴 교체 탐지기는 아직 학습 중이라, 이번에는 AI 생성 영상 탐지기만 사용했습니다.",
        "foot": "점수는 16프레임의 중앙값이며, 임계값을 넘으면 의심으로 판정합니다. 참고용입니다.",
        "cap_aivid": "AI영상", "cap_swap": "교체",
        "no_res": "아직 학습 결과가 없습니다.", "tr_seen": " (학습함)", "tr_unseen": " (처음 봄)",
        "tbl_head": "| 모델 | 학습한 가짜 | 실제 사진 정확도 | {cols} |",
        "tbl_foot": "테스트 세트는 학습에 나오지 않은 인물입니다 (IMDB-WIKI 폴더 90–99).",
        "gens": {"text2img": "SD 텍스트→이미지", "inpainting": "SD 인페인팅", "insight": "InsightFace 얼굴 교체"},
    },
}


def run_names():
    return sorted(p.name for p in RUNS.glob("*") if (p / "model.pt").exists())


@lru_cache(maxsize=8)
def load(name):
    m = build_model(pretrained=False)
    if name not in run_names():
        raise gr.Error("unknown model")
    m.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu", weights_only=True))
    return m.to(DEVICE).eval()


@lru_cache(maxsize=2)
def load_b4(ckpt):
    """Tight-crop EfficientNet-B4 detector + its validation threshold, or None if not trained yet."""
    if not Path(ckpt).exists():
        return None
    from train_swap import build as build_b4

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = build_b4(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(DEVICE).eval(), ck["threshold"]


def square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256), Image.BICUBIC)


def analyze(img, run, lang="zh"):
    t = TXT[lang]
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
    md = t["img_md"].format(v=t["v_fake_img"] if p > 0.5 else t["v_real_img"], p=p, m=run)
    return {t["fake"]: p, t["real"]: 1 - p}, overlay, md


@torch.no_grad()
def analyze_video(path, lang="zh"):
    import cv2

    from extract_tight import tight_crop, video_frames
    from video_scan import faces_in

    t = TXT[lang]
    frames = video_frames(path, 16)
    rows, gallery = [], []

    tight = None  # tight 380 px face crops, shared by both B4 detectors

    def b4_scores(det, cap):
        nonlocal tight
        from train_swap import eval_tf as b4_tf

        if tight is None:
            tight = []
            for _, f in frames:
                c = tight_crop(f)
                if c is not None:
                    tight.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
        m, thr = det
        ps = []
        for im in tight:
            p = torch.sigmoid(m(b4_tf(im).unsqueeze(0).to(DEVICE))).item()
            ps.append(p)
            if len(ps) % 3 == 1:
                gallery.append((im, f"{cap} {p:.2f}"))
        return ps, thr

    # 1) AI-generated video: B4 on tight crops if trained, else the loose-crop image-style model.
    aivid = load_b4(str(AIVID_CKPT))
    if aivid:
        ps, thr = b4_scores(aivid, t["cap_aivid"])
        if ps:
            rows.append((t["det_aivid"], float(np.median(ps)), thr, len(ps)))
    elif VIDEO_MODEL in run_names():
        m, ps = load(VIDEO_MODEL), []
        for _, f in frames:
            faces = faces_in(f)
            if faces:
                crop = max(faces, key=lambda r: r[0][2] * r[0][3])[1]
                p = torch.sigmoid(m(tf(crop).unsqueeze(0).to(DEVICE))).item()
                ps.append(p)
                if len(ps) % 2:
                    gallery.append((crop, f"{t['cap_aivid']} {p:.2f}"))
        if ps:
            rows.append((t["det_aivid"], float(np.median(ps)), 0.5, len(ps)))

    # 2) Face swap: B4 on tight crops.
    swap = load_b4(str(SWAP_CKPT))
    if swap:
        ps, thr = b4_scores(swap, t["cap_swap"])
        if ps:
            rows.append((t["det_swap"], float(np.median(ps)), thr, len(ps)))

    if not rows:
        return None, None, t["no_face"], []
    flagged = [r for r in rows if r[1] > r[2]]
    verdict = t["v_fake"].format(k=max(flagged, key=lambda r: r[1] - r[2])[0]) if flagged else t["v_real"]
    table = t["th"] + "\n|---|---|---|---|---|\n" + "\n".join(
        f"| {name} | {p:.2f} | {thr:.2f} | {n} | {t['r_susp'] if p > thr else t['r_ok']} |" for name, p, thr, n in rows)
    if not swap:
        table += "\n\n" + t["no_swap"]
    top = max(rows, key=lambda r: r[1] - r[2])
    md = verdict + "\n\n" + table + "\n\n" + t["foot"]
    return {t["susp"]: top[1], t["ok"]: 1 - top[1]}, None, md, gallery


def analyze_any(file, lang_name="中文"):
    lang = LANGS.get(lang_name, "zh")
    if file is None:
        return None, None, "", []
    path = Path(file)
    if path.suffix.lower() in VIDEO_EXT:
        return analyze_video(str(path), lang)
    try:
        img = Image.open(path)
    except Exception:
        return None, None, TXT[lang]["bad_file"], []
    lbl, cam, md = analyze(img, image_model(), lang)
    return lbl, cam, md, []


def analyze_pick(img, run, lang_name="中文"):
    if run is not None and run not in run_names():  # whitelist: only trained models in runs/
        raise gr.Error("unknown model")
    return analyze(img, run, LANGS.get(lang_name, "zh"))


def results_table(lang="zh"):
    t = TXT[lang]
    rows = []
    for name in run_names():
        f = RUNS / name / "results.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        trained = " + ".join(t["gens"][s] for s in r["fakes"])
        cells = []
        for s in FAKES:
            mark = t["tr_seen"] if s in r["fakes"] else t["tr_unseen"]
            cells.append(f"{r['test'][s]['auc']:.3f} / {r['test'][s]['acc']:.1%}{mark}")
        rows.append(f"| `{name}` | {trained} | {r['test']['real_acc']:.1%} | " + " | ".join(cells) + " |")
    if not rows:
        return t["no_res"]
    head = t["tbl_head"].format(cols=" | ".join(t["gens"][s] + " AUC / Acc" for s in FAKES)) + "\n"
    head += "|---" * (3 + len(FAKES)) + "|\n"
    return head + "\n".join(rows) + "\n\n" + t["tbl_foot"]


with gr.Blocks(title="AI Face Detector", delete_cache=(3600, 3600)) as demo:
    lang = gr.Radio(list(LANGS), value="中文", label="Language / 언어 / 语言")
    title = gr.Markdown(TXT["zh"]["title"])
    with gr.Tab(TXT["zh"]["tab_auto"]) as tab_auto:
        with gr.Row():
            with gr.Column():
                up = gr.File(label=TXT["zh"]["upload"], type="filepath", file_types=["image", "video", ".heic", ".heif"])
                go = gr.Button(TXT["zh"]["detect"], variant="primary")
            with gr.Column():
                umd = gr.Markdown()
                ulbl = gr.Label(label=TXT["zh"]["prob"])
                ucam = gr.Image(label=TXT["zh"]["cam"], height=320)
                ugal = gr.Gallery(label=TXT["zh"]["gallery"], columns=4, height=320)
        go.click(analyze_any, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any", concurrency_id="detector", concurrency_limit=1)
        up.upload(analyze_any, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any_upload", concurrency_id="detector", concurrency_limit=1)
    with gr.Tab(TXT["zh"]["tab_pick"]) as tab_pick:
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label=TXT["zh"]["img_in"], height=360)
                run = gr.Dropdown(run_names(), label=TXT["zh"]["model"], allow_custom_value=False)
                btn = gr.Button(TXT["zh"]["detect"], variant="primary")
            with gr.Column():
                md = gr.Markdown()
                lbl = gr.Label(label=TXT["zh"]["prob"])
                cam = gr.Image(label=TXT["zh"]["cam_pick"], height=360)
        btn.click(analyze_pick, [inp, run, lang], [lbl, cam, md], concurrency_id="detector", concurrency_limit=1)
        inp.change(analyze_pick, [inp, run, lang], [lbl, cam, md], concurrency_id="detector", concurrency_limit=1)
    with gr.Tab(TXT["zh"]["tab_arms"]) as tab_arms:
        arms_intro = gr.Markdown(TXT["zh"]["arms_intro"])
        table = gr.Markdown()

    def relabel(lang_name):
        t = TXT[LANGS[lang_name]]
        # Old results are cleared (value=None/"") so they do not stay in the previous language.
        return (gr.update(value=t["title"]), gr.update(label=t["tab_auto"]), gr.update(label=t["tab_pick"]),
                gr.update(label=t["tab_arms"]), gr.update(label=t["upload"]), gr.update(value=t["detect"]),
                gr.update(label=t["prob"], value=None), gr.update(label=t["cam"], value=None),
                gr.update(label=t["gallery"], value=None), gr.update(label=t["img_in"]),
                gr.update(label=t["model"]), gr.update(value=t["detect"]),
                gr.update(label=t["prob"], value=None), gr.update(label=t["cam_pick"], value=None),
                gr.update(value=t["arms_intro"]), gr.update(value=results_table(LANGS[lang_name])),
                gr.update(value=""), gr.update(value=""))

    lang.change(relabel, lang, [title, tab_auto, tab_pick, tab_arms, up, go, ulbl, ucam, ugal,
                                inp, run, btn, lbl, cam, arms_intro, table, umd, md])

    def refresh():
        names = run_names()
        value = image_model()
        return gr.update(choices=names, value=value), results_table("zh")

    demo.load(refresh, None, [run, table])

if __name__ == "__main__":
    demo.launch()
