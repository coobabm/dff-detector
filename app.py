"""Web UI for AI-face detection (中文 / English / 한국어).

  uv run app.py   ->  http://127.0.0.1:7860

"Detect" tab: images -> best image model; videos -> AI-video v4 + face swap. Per-model scores and thresholds.
Disagreements and failures are explicit. Below-threshold scores do not prove authenticity.
"Pick a model" tab: run one B0 detector with Grad-CAM.
"""
import html
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
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

import sys  # noqa: E402
sys.path[:0] = [str(Path(__file__).resolve().parent / d) for d in ("detector", "training", "data_prep", "evaluation")]

from train import FAKES, MEAN, STD, build_model

register_heif_opener()  # iPhone HEIC/HEIF uploads

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
VIDEO_MODEL = "all_video"  # fallback AI-video detector (loose crops) if the B4 one is not trained
SWAP_CKPT = ROOT / "runs_swap" / "b4" / "model.pt"
AIVID_CKPT = ROOT / "runs_aivid" / "b4" / "model.pt"


def image_model():
    """Best image model available: all_shot (+ AI-video screenshots), all_image (AI images + StyleGAN), all_openai."""
    for name in ("all_shot", "all_image", "all_openai"):
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
        "c_det": "检测器", "c_score": "分数", "c_thr": "阈值", "c_frames": "帧", "p_ai": "AI 生成概率",
        "empty": "上传一张含人脸的图片或一段视频，结果会显示在这里。",
        "sub": "上传含人脸的图片或视频，自动交给对应的检测模型。",
        "note": "模型分数不是鉴定结论，仅供参考。",
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
        "c_det": "Detector", "c_score": "Score", "c_thr": "Threshold", "c_frames": "Frames", "p_ai": "AI-generated probability",
        "empty": "Upload an image or video with a face. The report appears here.",
        "sub": "Upload an image or video with a face; it goes to the matching detector automatically.",
        "note": "Scores are not proof of authenticity. For reference only.",
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
        "c_det": "탐지기", "c_score": "점수", "c_thr": "임계값", "c_frames": "프레임", "p_ai": "AI 생성 확률",
        "empty": "얼굴이 있는 이미지나 영상을 올리면 여기에 결과가 표시됩니다.",
        "sub": "얼굴이 있는 이미지나 영상을 올리면 알맞은 탐지 모델로 자동 판별합니다.",
        "note": "모델 점수는 진위 판정의 근거가 아니며 참고용입니다.",
        "no_res": "아직 학습 결과가 없습니다.", "tr_seen": " (학습함)", "tr_unseen": " (처음 봄)",
        "tbl_head": "| 모델 | 학습한 가짜 | 실제 사진 정확도 | {cols} |",
        "tbl_foot": "테스트 세트는 학습에 나오지 않은 인물입니다 (IMDB-WIKI 폴더 90–99).",
        "gens": {"text2img": "SD 텍스트→이미지", "inpainting": "SD 인페인팅", "insight": "InsightFace 얼굴 교체"},
    },
}


def esc(x):
    return html.escape(str(x))


def bare(heading):
    """'### 판정: AI 조작 의심' -> 'AI 조작 의심' (the report shows the verdict word alone)."""
    text = heading.lstrip("# ").strip()
    for sep in ("：", ": "):
        if sep in text:
            text = text.split(sep, 1)[1]
    return text[:1].upper() + text[1:]


def meter(p, thr):
    """Horizontal score bar with a tick at the decision threshold."""
    kind = "fake" if p > thr else "real"
    return (f'<div class="meter {kind}" role="img" aria-label="{p:.2f} / {thr:.2f}">'
            f'<i style="width:{p * 100:.1f}%"></i><b style="left:{thr * 100:.1f}%"></b></div>')


STATUS = {"zh": {"fake": "可疑", "real": "正常"}, "en": {"fake": "Suspicious", "real": "Normal"},
          "ko": {"fake": "의심", "real": "정상"}}


def report(kind, verdict, detail="", rows=None, t=None, note=""):
    """Verdict strip. kind: fake | real | empty | error | loading. Left: status + verdict; right: per-detector evidence."""
    status = ""
    if kind in ("fake", "real") and t is not None:
        code = next(c for c, v in TXT.items() if v is t)
        status = f'<span class="status">{esc(STATUS[code][kind])}</span>'
    body = f'<div class="lead">{status}<div class="verdict">{esc(verdict)}</div>'
    if detail:
        body += f'<p class="detail">{detail}</p>'
    body += '</div><div class="evidence">'
    if rows:
        body += (f'<table class="dets"><thead><tr><th>{esc(t["c_det"])}</th><th>{esc(t["c_score"])}</th>'
                 f'<th class="num">{esc(t["c_thr"])}</th><th class="num">{esc(t["c_frames"])}</th><th>{esc(ALL_TEXT[next(c for c, v in TXT.items() if v is t)]["result"])}</th></tr></thead><tbody>')
        lang = next(c for c, v in TXT.items() if v is t)
        for name, p, thr, n in rows:
            if p is None:
                body += f'<tr><td>{esc(name)}</td><td>—</td><td>—</td><td>0</td><td>{esc(ALL_TEXT[lang]["incomplete"])}</td></tr>'
                continue
            body += (f'<tr class="{"fake" if p > thr else "real"}"><td>{esc(name)}</td>'
                     f'<td>{meter(p, thr)}<span class="score">{p:.2f}</span></td>'
                     f'<td class="num">{thr:.3f}</td><td class="num">{n}</td><td>{esc(ALL_TEXT[lang]["above" if p > thr else "below"])}</td></tr>')
        body += "</tbody></table>"
    body += "</div>"
    if note:
        body += f'<p class="note">{esc(note)}</p>'
    layout = "" if rows else " single"
    return f'<section class="report {kind}{layout}" aria-live="polite">{body}</section>'


def run_names():
    return sorted(p.name for p in RUNS.glob("*") if (p / "model.pt").exists())


# One lock for both caches: the preload thread and a request never load the same file twice.
_LOAD_LOCK = threading.Lock()


def ensemble():
    """runs/fm_ens/ensemble.json: {"members": [...], "threshold": t} -- average of fine-tuned ViT detectors."""
    f = RUNS / "fm_ens" / "ensemble.json"
    if not f.exists():
        return None
    e = json.loads(f.read_text())
    return e if all((RUNS / m / "model.pt").exists() for m in e["members"]) else None


def arch_of(name):
    """runs/<name>/arch.json for timm ViT detectors (fm_detector.py); None for the EfficientNet-B0 ones."""
    f = RUNS / name / "arch.json"
    return json.loads(f.read_text()) if f.exists() else None


@lru_cache(maxsize=16)
def image_tf(name):
    a = arch_of(name)
    if not a:
        return tf
    return T.Compose([T.Resize(a["size"]), T.CenterCrop(a["size"]), T.ToTensor(), T.Normalize(a["mean"], a["std"])])


@lru_cache(maxsize=16)
def _load(name):
    if name not in run_names():
        raise gr.Error("unknown model")
    a = arch_of(name)
    if a:
        import timm

        m = timm.create_model(a["timm"], pretrained=False, num_classes=1, img_size=a["size"])
    else:
        m = build_model(pretrained=False)
    m.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu", weights_only=True))
    return m.to(DEVICE).eval()


def load(name):
    with _LOAD_LOCK:
        return _load(name)


@lru_cache(maxsize=16)
def _load_b4(ckpt):
    if not Path(ckpt).exists():
        return None
    from train_swap import build as build_b4

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = build_b4(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(DEVICE).eval(), ck["threshold"]


def load_b4(ckpt):
    """Tight-crop EfficientNet-B4 detector + its validation threshold, or None if not trained yet."""
    with _LOAD_LOCK:
        return _load_b4(ckpt)


def load_ext(key):
    import external_detectors

    with _LOAD_LOCK:
        return external_detectors.load(key, DEVICE)


WARM = {"done": 0, "total": 0, "started": False}


def preload():
    """Load every checkpoint in the background, one by one, so the first upload does not wait for all of them."""
    specs = model_specs()
    WARM["total"] = len(specs)
    for name, family, key in specs:
        try:
            if family == "ext":
                load_ext(key)
            elif family == "ens":
                for m in ensemble()["members"]:
                    load(m)
            else:
                load_b4(key) if family == "b4" else load(key)
        except Exception:
            logging.exception("Preload failed: %s", name)
        WARM["done"] += 1
    logging.info("Preloaded %d models on %s", len(specs), DEVICE)


def start_preload():
    if not WARM["started"]:
        WARM["started"] = True
        threading.Thread(target=preload, name="preload", daemon=True).start()


def b0_threshold(name):
    """Decision threshold of an image model: runs/<name>/threshold.json if present, else 0.5."""
    f = RUNS / name / "threshold.json"
    return float(json.loads(f.read_text())["threshold"]) if f.exists() else 0.5


def square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256), Image.BICUBIC)


def grad_cam(model, im, transform):
    """Explain the fake-class logit on the exact model input, without parameter gradients."""
    x = transform(im).unsqueeze(0).to(DEVICE)
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        # ViT: the logit reads only the CLS token after the last block, so final patch tokens get zero
        # gradient. Explain the tokens entering the last block instead (they still feed CLS via attention).
        store = {}
        hook = model.blocks[-2].register_forward_hook(lambda mod, inp, out: store.__setitem__("t", out))
        try:
            with torch.enable_grad():
                logit = model(x).squeeze()
                grads, = torch.autograd.grad(logit, store["t"])
        finally:
            hook.remove()
        n = getattr(model, "num_prefix_tokens", 1)
        f, g = store["t"].detach()[:, n:], grads[:, n:]
        side = int(round(f.shape[1] ** .5))
        # Per-token gradient x activation (channel-averaged Grad-CAM is dominated by DINOv2's few huge channels).
        rel = (g * f).sum(-1)
        cam = F.relu(rel) if rel.max() > 0 else rel.abs()
        cam = (cam / (cam.max() + 1e-30)).reshape(1, 1, side, side)  # gradients are tiny: normalize here
        features = None
    else:
        with torch.no_grad():
            features = model.forward_features(x)
        with torch.enable_grad():
            features = features.detach().requires_grad_(True)
            logit = model.forward_head(features).squeeze()
            grads, = torch.autograd.grad(logit, features)
    score = torch.sigmoid(logit.detach()).item()
    if features is None:
        pass
    else:
        cam = F.relu((grads.mean((2, 3), keepdim=True) * features.detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    # Use the normalized tensor's actual crop as the base (B0 center crop / B4 tight crop).
    norm = next((t for t in getattr(transform, "transforms", []) if isinstance(t, T.Normalize)), None)
    mean, std = (norm.mean, norm.std) if norm else (MEAN, STD)
    rgb = x[0].detach().cpu().permute(1, 2, 0).numpy() * np.asarray(std) + np.asarray(mean)
    base = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
    if not np.isfinite(cam).all() or not np.isfinite(score):
        raise ValueError("Non-finite Grad-CAM")
    if cam.max() <= 1e-8:
        return score, base  # no positive activation: do not invent colored evidence
    cam /= cam.max()
    heat = Image.fromarray((cm.jet(cam)[..., :3] * 255).astype(np.uint8))
    return score, Image.blend(base, heat, .45)


def analyze(img, run, lang="zh"):
    t = TXT[lang]
    if img is None or not run:
        return None, None, report("empty", "", "", t=t, note=t["empty"])
    model = load(run)
    im = square(ImageOps.exif_transpose(img).convert("RGB"))
    p, overlay = grad_cam(model, im, image_tf(run))
    thr = b0_threshold(run)
    detail = f'{esc(t["p_ai"])} <strong>{p:.1%}</strong> · <code>{esc(run)}</code>' + meter(p, thr)
    md = report("fake" if p > thr else "real", t["v_fake_img"] if p > thr else t["v_real_img"], detail,
                t=t, note=t["note"])
    return {t["fake"]: p, t["real"]: 1 - p}, overlay, md


ALL_TEXT = {
    "zh": dict(sub="图片交给 AI 图片检测器；视频交给 AI 生成视频和换脸两个检测器。低分不能证明真实。",
               mixed="模型意见不一致", flagged="模型提示可疑", low="均未超过阈值，但不能证明真实",
               incomplete="检测不完整", missing="没有可用人脸", failed="推理失败", result="结果",
               below="未超过阈值", above="超过阈值", count="{flagged}/{total} 个成功运行的模型超过各自阈值。",
               note="各模型分数未统一校准，不是鉴定概率，也不作投票结论。视频分数为最多 16 个采样帧的中位数；视频专用模型用于静态图片仅供探索性参考。"),
    "en": dict(sub="Images go to the AI-image detector; videos go to the AI-video and face-swap detectors. Low scores are not proof of authenticity.",
               mixed="Models disagree", flagged="Models flag suspicious content", low="Below all thresholds; authenticity is not proven",
               incomplete="Incomplete detection", missing="No usable face", failed="Inference failed", result="Result",
               below="Below threshold", above="Above threshold", count="{flagged}/{total} successful models exceed their own thresholds.",
               note="Scores are not jointly calibrated probabilities or a vote. Video scores are medians over up to 16 sampled frames. Video specialists on still images are exploratory only."),
    "ko": dict(sub="이미지는 AI 이미지 탐지기로, 영상은 AI 생성 영상·얼굴 교체 탐지기로 검사합니다. 낮은 점수가 진위를 증명하지 않습니다.",
               mixed="모델 의견 불일치", flagged="모델이 의심 신호를 감지함", low="모두 임계값 이하이나 진짜임을 증명하지 않음",
               incomplete="불완전한 탐지", missing="사용 가능한 얼굴 없음", failed="추론 실패", result="결과",
               below="임계값 이하", above="임계값 초과", count="성공한 모델 {total}개 중 {flagged}개가 각 임계값을 초과했습니다.",
               note="점수는 통합 보정된 확률이나 투표 결과가 아닙니다. 영상 점수는 최대 16개 프레임의 중앙값입니다. 영상 전용 모델의 정지 이미지 결과는 탐색적 참고용입니다."),
}
CAM_TEXT = {
    "zh": dict(title="多模型 Grad-CAM · 点击放大", image="当前图片", frame="采样帧",
               score="此图分数", threshold="阈值", failed="热力图生成失败", no_frame="未找到两种裁切均可用的同一帧，未生成视频热力图。",
               note="热力图解释各模型对 AI/篡改类别的响应，不是伪造区域证据。每张图独立归一化，红色强度不能跨模型比较；无正向响应时显示原裁切。B0/B4 输入裁切不同。视频热力图只解释标注帧，其分数不是整段视频的中位数。"),
    "en": dict(title="Multi-model Grad-CAM · click to enlarge", image="Current image", frame="Sampled frame",
               score="This crop score", threshold="Threshold", failed="Heatmap failed", no_frame="No shared frame with both crop types; video heatmaps unavailable.",
               note="Maps explain each model's fake-class response, not proof of manipulated regions. Each map is independently normalized; color intensity is not comparable across models. No positive response shows the original crop. B0/B4 use different crops. Video maps explain only the labeled frame, not the video's median score."),
    "ko": dict(title="다중 모델 Grad-CAM · 클릭하여 확대", image="현재 이미지", frame="샘플 프레임",
               score="이 이미지 점수", threshold="임계값", failed="히트맵 생성 실패", no_frame="두 크롭에 공통으로 사용할 프레임이 없어 영상 히트맵을 생성하지 않았습니다.",
               note="히트맵은 AI/조작 클래스에 대한 모델 반응이며 조작 영역의 증거가 아닙니다. 각 맵은 개별 정규화되어 모델 간 색 강도를 비교할 수 없습니다. 양의 반응이 없으면 원본 크롭을 표시합니다. B0/B4 크롭은 다릅니다. 영상 맵은 표시된 프레임만 설명하며 영상 중앙값 점수가 아닙니다."),
}
for _lang, _copy in ALL_TEXT.items():
    TXT[_lang]["sub"] = _copy["sub"]


# One detector per job: images, AI-generated video, face swap. Older versions stay on disk but are not run.
B4_MODELS = (("runs_aivid/v4", "runs_aivid/b4"), ("runs_swap/b4",))


def model_specs(kind=None):
    """Detectors for one-click detection. kind="image": best image model only (the video models
    flag about half of real still photos) + published detectors (external_detectors.py) on the original
    upload; kind="video": AI-video v4 + face swap; None: everything."""
    import external_detectors

    specs = []
    ens = ensemble()
    image = image_model()
    if kind != "video":
        if ens:  # fine-tuned CLIP + DINOv2 average replaces the B0 image model
            specs.append(("runs/fm_ens", "ens", "fm_ens"))
        elif image:
            specs.append((f"runs/{image}", "b0", image))
    if kind != "video":
        specs.extend((f"ext/{external_detectors.NAMES[k]}", "ext", k) for k in external_detectors.available() if k != "bfree")
    for choices in B4_MODELS if kind != "image" else ():
        name = next((n for n in choices if (ROOT / n / "model.pt").exists()), None)
        if name:
            specs.append((name, "b4", str(ROOT / name / "model.pt")))
    return specs


def score_all(loose, tight, lang, **kw):
    rows = []
    for rows in iter_scores(loose, tight, lang, **kw):
        pass
    return rows


DETECTOR_WORKERS = 3


def iter_scores(loose, tight, lang, *, heatmaps=None, cam_errors=None, specs=None, **kw):
    """Three concurrent detectors; publish scores immediately, before optional heatmaps.

    Workers own their output lists. Only the coordinator updates shared UI state.
    The outer Gradio queue remains single-request so hooks/cached models are not shared
    by concurrent requests. Cancellation drains workers before releasing that queue slot.
    """
    selected = list(model_specs() if specs is None else specs)
    if not selected:
        return
    events = Queue()

    def worker(index, spec):
        maps, errors = [], []
        emitted = False
        try:
            for result in _iter_scores_single(loose, tight, lang, specs=[spec],
                    heatmaps=maps if heatmaps is not None else None, cam_errors=errors, **kw):
                if result and not emitted:
                    events.put(("row", index, result[0]))
                    emitted = True
        except Exception as exc:
            logging.exception("Detector task failed: %s", spec[0])
            if not emitted:
                events.put(("row", index, (spec[0] + " · " + ALL_TEXT[lang]["failed"]
                            + " (" + type(exc).__name__ + ")", None, None, 0)))
            else:
                errors.append(spec[0])
        finally:
            events.put(("done", index, (maps, errors)))

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=DETECTOR_WORKERS, thread_name_prefix="detector") as pool:
        for index, spec in enumerate(selected):
            pool.submit(worker, index, spec)
        while done < len(selected):
            kind, index, payload = events.get()
            if kind == "row":
                rows.append(payload)
                yield list(rows)
            else:
                maps, errors = payload
                if heatmaps is not None:
                    heatmaps.extend(maps)
                if cam_errors is not None:
                    cam_errors.extend(errors)
                done += 1


@torch.no_grad()
def _iter_scores_single(loose, tight, lang, *, heatmaps=None, cam_errors=None, representatives=None, frame_label="",
                specs=None, originals=None):
    """Run the detectors one at a time; yield the rows so far after each one.
    b0: loose face crops, b4: tight crops, ext (published detectors): the original uploaded image(s)."""
    from train_swap import eval_tf as b4_tf

    rows = []
    for name, family, key in specs or model_specs():
        if family == "ext":
            import external_detectors

            try:
                load_ext(key)
                scores = [external_detectors.score(key, im.convert("RGB"), DEVICE) for im in originals or []]
                if not scores:
                    rows.append((name + " · " + ALL_TEXT[lang]["missing"], None, None, 0))
                elif not np.isfinite(scores).all():
                    raise ValueError("Non-finite score")
                else:
                    rows.append((name, float(np.median(scores)), external_detectors.THRESHOLDS[key], len(scores)))
            except Exception as exc:
                logging.exception("Detector failed: %s", name)
                rows.append((name + " · " + ALL_TEXT[lang]["failed"] + " (" + type(exc).__name__ + ")", None, None, 0))
            yield list(rows)
            continue
        if family == "ens":
            try:
                e = ensemble()
                if not loose:
                    rows.append((name + " · " + ALL_TEXT[lang]["missing"], None, None, 0))
                    yield list(rows)
                    continue
                one = heatmaps is not None and bool(representatives) and len(loose) == 1 \
                    and representatives.get("b0") is loose[0]
                maps = []
                if one:
                    # Single image: each member's Grad-CAM pass already yields its score, so there is no
                    # separate scoring forward (same numbers; about half the time on a CPU server).
                    per = [[]]
                    for m in e["members"]:
                        try:
                            score, overlay = grad_cam(load(m), loose[0], image_tf(m))
                            if e.get("tta"):  # test-time augmentation: average with the mirrored crop
                                with torch.no_grad():
                                    flipped = image_tf(m)(ImageOps.mirror(loose[0])).unsqueeze(0).to(DEVICE)
                                    score = (score + torch.sigmoid(load(m)(flipped)).item()) / 2
                            maps.append((overlay, f"{m} · {frame_label} · ViT 224×224 · {CAM_TEXT[lang]['score']} {score:.3f}"))
                        except Exception:
                            if cam_errors is not None:
                                cam_errors.append(m)
                            with torch.no_grad():
                                score = torch.sigmoid(load(m)(image_tf(m)(loose[0]).unsqueeze(0).to(DEVICE))).item()
                        per[0].append(score)
                else:
                    with torch.no_grad():
                        per = [[float(np.mean([torch.sigmoid(load(m)(image_tf(m)(v).unsqueeze(0).to(DEVICE))).item()
                                               for v in ([im, ImageOps.mirror(im)] if e.get("tta") else [im])]))
                                for m in e["members"]] for im in loose]
                scores = [float(np.mean(p)) for p in per]
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite score")
                rows.append((name, float(np.median(scores)), float(e["threshold"]), len(scores)))
                if maps:
                    heatmaps.extend(maps)
                yield list(rows)  # score must not wait for Grad-CAM
                if heatmaps is not None and representatives and "b0" in representatives and not one:
                    for m in e["members"]:
                        try:
                            score, overlay = grad_cam(load(m), representatives["b0"], image_tf(m))
                            words = CAM_TEXT[lang]
                            heatmaps.append((overlay, f"{m} · {frame_label} · ViT 224×224 · {words['score']} {score:.3f}"))
                        except Exception:
                            if cam_errors is not None:
                                cam_errors.append(m)
            except Exception as exc:
                logging.exception("Detector failed: %s", name)
                rows.append((name + " · " + ALL_TEXT[lang]["failed"] + " (" + type(exc).__name__ + ")", None, None, 0))
            yield list(rows)
            continue
        crops = tight if family == "b4" else loose
        if not crops:
            rows.append((name + " · " + ALL_TEXT[lang]["missing"], None, None, 0))
            yield list(rows)
            continue
        try:
            model, threshold = load_b4(key) if family == "b4" else (load(key), b0_threshold(key))
            transform = b4_tf if family == "b4" else image_tf(key)
            scores = [torch.sigmoid(model(transform(im).unsqueeze(0).to(DEVICE))).item() for im in crops]
            if not np.isfinite(scores).all() or not np.isfinite(threshold):
                raise ValueError("Non-finite score or threshold")
            rows.append((name, float(np.median(scores)), float(threshold), len(scores)))
            yield list(rows)  # score must not wait for Grad-CAM
            if heatmaps is not None and representatives and family in representatives:
                try:
                    score, overlay = grad_cam(model, representatives[family], transform)
                    words = CAM_TEXT[lang]
                    crop_label = "B4 tight 380×380" if family == "b4" else ("ViT 224×224" if arch_of(key) else "B0 224×224")
                    caption = (f"{name} · {frame_label} · {crop_label} · "
                               f"{words['score']} {score:.3f} · {words['threshold']} {threshold:.3f}")
                    heatmaps.append((overlay, caption))
                except Exception:
                    if cam_errors is not None:
                        cam_errors.append(name)

        except Exception as exc:
            # A broken checkpoint must not hide the other detectors or become a 'real' vote.
            logging.exception("Detector failed: %s", name)
            rows.append((name + " · " + ALL_TEXT[lang]["failed"] + " (" + type(exc).__name__ + ")", None, None, 0))
        yield list(rows)


TRAINING_TYPES = {
    "runs/t2i": ("AI 文生图（SD 类）", "AI text-to-image (SD family)", "AI 텍스트 이미지 생성(SD 계열)"),
    "runs/inpaint": ("AI 局部重绘", "AI inpainting", "AI 인페인팅"),
    "runs/insight": ("AI 换脸", "AI face swapping", "AI 얼굴 교체"),
    "runs/all": ("混合类型 AI 生图／人脸篡改", "mixed AI image generation / face manipulation", "혼합 유형 AI 이미지 생성 / 얼굴 조작"),
    "runs/all_openai": ("混合类型 AI 生图／篡改（含 OpenAI 生图训练样本）", "mixed AI generation / manipulation (trained with OpenAI image samples)", "혼합 유형 AI 생성 / 조작(OpenAI 이미지 학습 샘플 포함)"),
    "runs/all_image": ("混合类型 AI 生图／篡改（含 GAN）", "mixed AI image generation / manipulation (including GANs)", "혼합 유형 AI 이미지 생성 / 조작(GAN 포함)"),
    "runs/fm_ens": ("AI 生成／篡改人脸（含新一代生成器，CLIP+DINOv2）", "AI-generated / manipulated face (incl. newest generators, CLIP+DINOv2)", "AI 생성 / 조작 얼굴(최신 생성기 포함, CLIP+DINOv2)"),
    "runs/all_shot": ("混合类型 AI 生图／篡改（含 GAN、AI 视频截图）", "mixed AI image generation / manipulation (including GANs and AI-video screenshots)", "혼합 유형 AI 이미지 생성 / 조작(GAN, AI 영상 캡처 포함)"),
    "runs/all_video": ("混合类型 AI 图片／视频生成或篡改", "mixed AI image / video generation or manipulation", "혼합 유형 AI 이미지 / 영상 생성 또는 조작"),
    "runs_swap": ("AI 换脸", "AI face swapping", "AI 얼굴 교체"),
    "ext": ("AI 生成图像（外部公开模型）", "AI-generated image (published detector)", "AI 생성 이미지(공개 탐지 모델)"),
    "runs_aivid": ("AI 视频生成／人像驱动", "AI video generation / portrait animation", "AI 영상 생성 / 인물 애니메이션"),
}
HINT_TEXT = {
    "zh": dict(prefix="疑似 {kind}", source="最高分模型（并列时全部列出）：{models}；分数 {score:.4f} > 0.9。",
               note="类型提示仅按最高分模型的训练范围生成，不是来源识别；不能据此确定具体生成工具，分数也不代表真实性概率。",
               unknown="AI 生成／篡改（训练类型未标注）"),
    "en": dict(prefix="Suspected {kind}", source="Highest-scoring model(s), including ties: {models}; score {score:.4f} > 0.9.",
               note="Type hints reflect the highest-scoring model's training scope, not provenance identification. They cannot identify the generating tool; scores are not authenticity probabilities.",
               unknown="AI generation / manipulation (training type unspecified)"),
    "ko": dict(prefix="{kind} 의심", source="최고 점수 모델(동점 포함): {models}; 점수 {score:.4f} > 0.9.",
               note="유형 안내는 최고 점수 모델의 학습 범위에 기반하며 출처 식별이 아닙니다. 실제 생성 도구를 확정할 수 없고 점수는 진위 확률이 아닙니다.",
               unknown="AI 생성 / 조작(학습 유형 미지정)"),
}


def training_hint(valid, lang):
    """UI heuristic only: never attribute a binary detector's score to a specific tool."""
    if not valid:
        return None
    best = max(r[1] for r in valid)
    if best <= .9:
        return None
    winners = [r for r in valid if r[1] == best]
    index = {"zh": 0, "en": 1, "ko": 2}[lang]
    text = HINT_TEXT[lang]
    kinds = []
    for name, *_ in winners:
        labels = TRAINING_TYPES.get(name) or TRAINING_TYPES.get(name.split("/", 1)[0])
        kind = labels[index] if labels else text["unknown"]
        if kind not in kinds:
            kinds.append(kind)
    return (text["prefix"].format(kind=" / ".join(kinds)),
            text["source"].format(models=", ".join(r[0] for r in winners), score=best))


RATE_TEXT = {
    "zh": dict(rate="AI 率 {r}（组合模型，按真假各半估计）。", label="AI 率", sub="组合模型，按真假各半估计",
               note="AI 率由组合模型分数在留出数据上校准得到（保序回归，假设送检图片真假各半）；"
                    "如果平时 AI 图更少，实际 AI 率会更低。判定仍以阈值为准。"),
    "en": dict(rate="AI rate {r} (ensemble, assuming half of images are AI). ", label="AI rate",
               sub="ensemble, assuming half of images are AI",
               note="The AI rate is the ensemble score calibrated on held-out data (isotonic regression, 50/50 prior); "
                    "if AI images are rarer among your uploads, the true rate is lower. The verdict still uses the threshold."),
    "ko": dict(rate="AI 확률 {r} (조합 모델, 진짜·AI 반반 가정). ", label="AI 확률", sub="조합 모델, 진짜·AI 반반 가정",
               note="AI 확률은 조합 모델 점수를 보류 데이터로 보정한 값입니다(등위 회귀, 반반 사전확률). "
                    "실제로 AI 이미지가 더 드물면 실제 확률은 더 낮습니다. 판정은 여전히 임계값 기준입니다."),
}


@lru_cache(maxsize=1)
def calibration():
    f = RUNS / "fm_ens" / "calibration.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    e = ensemble()
    return c if e and c.get("members") == e["members"] else None  # stale curve for other members: do not use


def ai_rate(score):
    c = calibration()
    return None if c is None else float(np.interp(score, c["x"], c["y"]))


def all_report(rows, lang, cam_errors=None, extra_note=""):
    text = ALL_TEXT[lang]
    valid = [r for r in rows if r[1] is not None]
    flagged = [r for r in valid if r[1] > r[2]]
    verdict = text["mixed"] if flagged and len(flagged) < len(valid) else text["flagged"] if flagged else text["low"]
    hint = training_hint(valid, lang)
    detail = text["count"].format(flagged=len(flagged), total=len(valid))
    ens_row = next((r for r in valid if r[0] == "runs/fm_ens"), None)
    rate = ai_rate(ens_row[1]) if ens_row else None
    if rate is not None:
        extra_note = RATE_TEXT[lang]["note"] + " " + extra_note
    if hint:
        detail = hint[1] + " " + verdict + "。 " + detail
        verdict = hint[0]
    rate_html = ""
    if rate is not None:
        shown = ">99%" if rate >= .995 else "<1%" if rate < .005 else f"≈ {rate:.0%}"  # never claim certainty
        t = RATE_TEXT[lang]
        level = min(4, int(rate * 5))  # 5 colour steps, green -> amber -> red (inline styles are stripped by Gradio)
        rate_html = (f'<span class="airate r{level}">'
                     f'<span class="airate-label">{esc(t["label"])}</span><b>{esc(shown)}</b>'
                     f'<small>{esc(t["sub"])}</small></span>')
    if len(valid) < len(rows) or not valid:
        verdict = text["incomplete"] + (" · " + verdict if valid else "")
    note = HINT_TEXT[lang]["note"] + " " + text["note"] + " " + CAM_TEXT[lang]["note"] + " " + extra_note
    if cam_errors:
        note += " " + CAM_TEXT[lang]["failed"] + ": " + ", ".join(cam_errors)
    return report("fake" if flagged or hint else "empty", verdict,
                  rate_html + esc(detail), rows, TXT[lang], note)


STREAM_TEXT = {
    "zh": dict(read="正在找人脸", frames="正在抽帧找人脸", found="已看 {seen} 帧，{faces} 帧有人脸",
               run="最多 3 个检测器并行运行", done="已完成 {done}/{total}", next="进行中：{name}",
               cold="首次启动，模型还在载入（{done}/{total}），会慢一些。", wait="每个模型跑完就会出现在右边。"),
    "en": dict(read="Looking for a face", frames="Sampling frames for faces", found="{seen} frames checked, {faces} with a face",
               run="Running up to 3 detectors in parallel", done="{done}/{total} done", next="In progress: {name}",
               cold="Models are still loading after startup ({done}/{total}); this run is slower.",
               wait="Each model's row appears on the right as soon as it finishes."),
    "ko": dict(read="얼굴을 찾는 중", frames="프레임에서 얼굴을 찾는 중", found="{seen}프레임 확인, 얼굴 {faces}프레임",
               run="최대 3개 탐지기 병렬 실행 중", done="{done}/{total} 완료", next="진행 중: {name}",
               cold="서버 시작 직후라 모델을 불러오는 중입니다 ({done}/{total}). 조금 느릴 수 있습니다.",
               wait="모델이 끝날 때마다 오른쪽에 결과가 한 줄씩 나타납니다."),
}


def progress(lang, step, rows=None, done=0, total=0, next_name="", extra=""):
    """Interim report while a run is streaming: step title, progress bar, rows finished so far."""
    words = STREAM_TEXT[lang]
    parts = [esc(extra)] if extra else []
    if total:
        parts.append(esc(words["done"].format(done=done, total=total))
                     + (" · " + esc(words["next"].format(name=next_name)) if next_name else ""))
    detail = " ".join(parts)
    if total:
        detail += f'<span class="meter progress"><i style="width:{done / total * 100:.1f}%"></i></span>'
    note = words["wait"]
    if WARM["started"] and WARM["done"] < WARM["total"]:
        note = words["cold"].format(done=WARM["done"], total=WARM["total"]) + " " + note
    html_ = report("loading", step, detail, rows or None, TXT[lang], note)
    if rows:  # mark the newest row so it fades in instead of the table appearing all at once
        head, sep, tail = html_.rpartition("<tr")
        html_ = head + sep + " data-new" + tail
    return html_.replace('class="report loading single"', 'class="report loading"') if total else html_


PACE = 0  # no artificial presentation delay


def stream_scores(loose, tight, lang, kind=None, pace=PACE, **kw):
    """iter_scores + one interim report per finished model. The last yielded rows are the final rows."""
    specs = model_specs(kind)
    names = [s[0] for s in specs]
    yield progress(lang, STREAM_TEXT[lang]["run"], [], 0, len(specs), ", ".join(names)), []
    shown = time.monotonic()
    for rows in iter_scores(loose, tight, lang, specs=specs, **kw):
        if pace > 0:
            time.sleep(max(0.0, pace - (time.monotonic() - shown)))
        done = len(rows)
        completed = {r[0].split(" · ", 1)[0] for r in rows}
        nxt = ", ".join(n for n in names if n not in completed)
        yield progress(lang, STREAM_TEXT[lang]["run"], rows, done, len(specs), nxt), rows
        shown = time.monotonic()
    if pace > 0:
        time.sleep(pace)


def last(steps):
    out = None
    for out in steps:
        pass
    return out


def analyze_all_image(img, lang):
    return last(stream_all_image(img, lang))


def analyze_video(path, lang="zh"):
    return last(stream_video(path, lang))


def face_crop(img):
    """Largest face as the loose crop the image models were trained on (DFF style, same as video frames);
    the centre square when no face is found (already-cropped portraits, tiny faces)."""
    import cv2
    from video_scan import faces_in

    faces = faces_in(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    if not faces:
        return square(img)
    box, crop = max(faces, key=lambda r: r[0][2] * r[0][3])
    # Crop only when it helps: the face is small in the frame (video screenshot, group shot) AND the crop
    # keeps real detail. Re-cropping framed portraits or blowing up tiny faces (blur) raises false alarms.
    side = 2.2 * max(box[2], box[3])  # faces_in crop size in original pixels
    return crop if box[2] < 0.3 * min(img.size) and side >= 224 else square(img)


def stream_all_image(img, lang):
    yield None, None, progress(lang, STREAM_TEXT[lang]["read"]), []
    img = ImageOps.exif_transpose(img).convert("RGB")
    loose = face_crop(img)
    tight = []  # images only run the B0 image model (see model_specs)
    heatmaps, errors = [], []
    reps = {"b0": loose}
    rows = []
    for md, rows in stream_scores([loose], tight, lang, "image", heatmaps=heatmaps, cam_errors=errors,
                                  representatives=reps, frame_label=CAM_TEXT[lang]["image"], originals=[img]):
        yield None, None, md, []
    yield None, heatmaps, all_report(rows, lang, errors), []


def stream_video(path, lang="zh"):
    import cv2
    from extract_tight import tight_crop, video_frames
    from video_scan import faces_in

    loose, tight, gallery = [], [], []
    reps, frame_label = {}, ""
    words = STREAM_TEXT[lang]
    yield None, None, progress(lang, words["frames"]), []
    for seen, (index, frame) in enumerate(video_frames(path, 16), 1):
        faces = faces_in(frame)
        if faces:
            crop = max(faces, key=lambda r: r[0][2] * r[0][3])[1]
            loose.append(crop)
            gallery.append((crop, str(index)))
        crop = tight_crop(frame)
        if crop is not None:
            tight.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if faces and not reps:
                reps = {"b0": loose[-1], "b4": tight[-1]}
                frame_label = f"{CAM_TEXT[lang]['frame']} #{index}"
        yield None, None, progress(lang, words["frames"], extra=words["found"].format(seen=seen, faces=len(loose))), list(gallery)
    heatmaps, errors = [], []
    rows = []
    for md, rows in stream_scores(loose, tight, lang, "video", heatmaps=heatmaps, cam_errors=errors,
                                  representatives=reps, frame_label=frame_label):
        yield None, None, md, gallery
    extra = "" if reps else CAM_TEXT[lang]["no_frame"]
    yield None, heatmaps, all_report(rows, lang, errors, extra), gallery


def ui_step(lbl, cam, md, gal):
    """Show the heatmap for images and the face-frame strip for videos, hide empty panels."""
    return lbl, gr.update(value=cam or None, visible=bool(cam)), md, gr.update(value=gal or None, visible=bool(gal))


def analyze_any_ui(file, lang_name="中文"):
    return ui_step(*analyze_any(file, lang_name))


def analyze_any_stream(file, lang_name="中文"):
    """Streaming UI handler: the report fills in while frames are read and each model finishes."""
    for step in stream_any(file, lang_name):
        yield ui_step(*step)


def analyze_pick_ui(img, run, lang_name="中文"):
    lbl, cam, md = analyze_pick(img, run, lang_name)
    return lbl, gr.update(value=cam, visible=cam is not None), md


def analyze_any(file, lang_name="中文"):
    return last(stream_any(file, lang_name))


def stream_any(file, lang_name="中文"):
    lang = LANGS.get(lang_name, "zh")
    if file is None:
        yield None, None, report("empty", "", "", t=TXT[lang], note=TXT[lang]["empty"]), []
        return
    path = Path(file)
    if path.suffix.lower() in VIDEO_EXT:
        yield from stream_video(str(path), lang)
        return
    try:
        img = Image.open(path)
    except Exception:
        head, _, rest = TXT[lang]["bad_file"].partition("\n\n")
        yield None, None, report("error", bare(head), esc(rest), t=TXT[lang]), []
        return
    with img:
        yield from stream_all_image(img, lang)


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
    return head + "\n".join(rows) + "\n\n" + t["tbl_foot"] + arms_race_table(lang)


ARMS = {
    "zh": dict(title="### 全部模型 × 留出测试集（2026-10）", real="真人判对", fake="AI 检出", set="测试集",
               note="所有模型用同一个工作点：阈值取各自在一半留出真照片上的 97 分位（其余数字在另一半和其它集合上）。"
                    "真人行 = 判为真的比例（越高越少冤枉），AI 行 = 检出比例。粗体为该行最高。"
                    "“没见过”= 该模型训练时没有这种生成器。外部公开模型较慢，每个集合只抽 {cap} 张。",
               unseen="（所有模型都没见过）"),
    "en": dict(title="### All models × held-out sets (2026-10)", real="real judged real", fake="AI caught", set="Test set",
               note="Same operating point for every model: threshold = its 97th percentile on half of the held-out real photos "
                    "(numbers on the other half and on all other sets). Real rows = share judged real (higher = fewer false alarms); "
                    "AI rows = share caught. Bold = best in the row. Published detectors are slow: {cap} images per set.",
               unseen=" (unseen by every model)"),
    "ko": dict(title="### 전체 모델 × 보류 테스트 세트 (2026-10)", real="진짜로 판정", fake="AI 검출", set="테스트 세트",
               note="모든 모델을 같은 기준으로 비교: 보류한 실제 사진 절반에서 각 모델 점수의 97 백분위를 임계값으로 사용 "
                    "(수치는 나머지 절반과 다른 세트). 실제 행 = 진짜로 판정한 비율(높을수록 오탐 적음), AI 행 = 검출 비율. "
                    "굵게 = 행 최고. 공개 탐지기는 느려서 세트당 {cap}장만 사용.",
               unseen=" (모든 모델이 처음 봄)"),
}
SET_NAMES = {  # zh, en, ko
    "photo_real": ("真照片（6 个来源）", "real photos (6 sources)", "실제 사진(6개 출처)"),
    "video_real_frames": ("真视频截图", "real video frames", "실제 영상 프레임"),
    "fairface": ("FairFace（各年龄 / 肤色）", "FairFace (all ages / ethnicities)", "FairFace(전 연령/인종)"),
    "openfake_real": ("LAION / Pexels 真照片", "LAION / Pexels reals", "LAION / Pexels 실제"),
    "reddit_real": ("Reddit 摄影版", "Reddit photography", "Reddit 사진 게시판"),
    "dff_text2img": ("SD 1.5 文生图（DFF）", "SD 1.5 text2img (DFF)", "SD 1.5 텍스트→이미지(DFF)"),
    "dff_inpainting": ("SD 局部重绘（DFF）", "SD inpainting (DFF)", "SD 인페인팅(DFF)"),
    "dff_insight": ("InsightFace 换脸（DFF）", "InsightFace swap (DFF)", "InsightFace 얼굴 교체(DFF)"),
    "openai": ("OpenAI gpt-image 人脸", "OpenAI gpt-image faces", "OpenAI gpt-image 얼굴"),
    "stylegan": ("StyleGAN", "StyleGAN", "StyleGAN"),
    "ai_video_frames": ("AI 视频截图（Veo / Sora / Kling / 数字人）", "AI-video frames (Veo / Sora / Kling / talking heads)", "AI 영상 프레임(Veo/Sora/Kling/디지털 휴먼)"),
    "swap_frames": ("换脸视频截图", "face-swap video frames", "얼굴 교체 영상 프레임"),
    "sfhq": ("SFHQ（FLUX / SDXL / DALL-E 3）", "SFHQ (FLUX / SDXL / DALL-E 3)", "SFHQ(FLUX/SDXL/DALL-E 3)"),
    "reddit_ai": ("Reddit AI 图（MJ / Flux / SD）", "Reddit AI art (MJ / Flux / SD)", "Reddit AI 이미지(MJ/Flux/SD)"),
    "openfake": ("OpenFake（gpt-image / Nano Banana / Imagen 4 / MJ 7 …）", "OpenFake (gpt-image / Nano Banana / Imagen 4 / MJ 7 …)", "OpenFake(gpt-image/Nano Banana/Imagen 4/MJ 7 …)"),
    "rapidata_unseen": ("最新生成器（Seedream 3 / Qwen / Imagen 4 / Recraft …）", "newest generators (Seedream 3 / Qwen / Imagen 4 / Recraft …)", "최신 생성기(Seedream 3/Qwen/Imagen 4/Recraft …)"),
}
MODEL_NAMES = [  # key, column label, sets it was trained on (None = all except rapidata)
    ("t2i", "t2i", {"dff_text2img"}), ("inpaint", "inpaint", {"dff_inpainting"}), ("insight", "insight", {"dff_insight"}),
    ("all", "all", {"dff_text2img", "dff_inpainting", "dff_insight"}),
    ("all_image", "all_image", {"dff_text2img", "dff_inpainting", "dff_insight", "openai", "stylegan"}),
    ("all_shot", "all_shot", {"dff_text2img", "dff_inpainting", "dff_insight", "openai", "stylegan", "ai_video_frames", "swap_frames"}),
    ("fm_ft_dinov2_b", "DINOv2-ft", None), ("fm_ft_clip_l14", "CLIP-ft", None), ("fm_ens", {"zh": "**线上组合**", "en": "**live combination**", "ko": "**현재 서비스 조합**"}, None),
    ("ext_univfd", "UnivFD", set()), ("ext_npr", "NPR", set()), ("ext_bfree", "B-Free", set()),
]


def arms_race_table(lang="zh"):
    f = ROOT / "results" / "arms_race.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    w, idx = ARMS[lang], {"zh": 0, "en": 1, "ko": 2}[lang]
    unseen_word = TXT[lang]["tr_unseen"].strip()
    cols = [(k, label[lang] if isinstance(label, dict) else label, seen) for k, label, seen in MODEL_NAMES if k in d["models"]]
    out = [f"\n\n{w['title']}\n", f"| {w['set']} | " + " | ".join(label for _, label, _ in cols) + " |",
           "|---" * (1 + len(cols)) + "|"]
    for group, sets in ((w["real"], d["real_sets"]), (w["fake"], d["fake_sets"])):
        out.append(f"| **{group}** |" + " |" * len(cols))
        for s in sets:
            vals = [d["models"][k].get(s) for k, _, _ in cols]
            best = max(v for v in vals if v is not None)
            cells = []
            for (k, _, seen), v in zip(cols, vals):
                cell = f"**{v:.0%}**" if v == best else f"{v:.0%}"
                if s in d["fake_sets"] and seen is not None and s not in seen and not k.startswith("ext_"):
                    cell += f" <sub>{unseen_word}</sub>"
                cells.append(cell)
            name = SET_NAMES[s][idx] + (w["unseen"] if s == "rapidata_unseen" else "")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("\n" + w["note"].format(cap=150))
    return "\n".join(out)


THEME = gr.themes.Base(
    primary_hue="yellow", neutral_hue="zinc", radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("IBM Plex Sans KR"), gr.themes.GoogleFont("Noto Sans SC"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#FFFFFF", body_text_color="#1D1F21", body_text_color_subdued="#5A6068",
    block_background_fill="#FFFFFF", block_border_color="#E2E5E8", block_border_width="1px", block_shadow="none",
    block_label_background_fill="#FFFFFF", block_label_text_color="#5A6068", block_title_text_color="#5A6068",
    border_color_primary="#E2E5E8", border_color_accent="#1D1F21", input_background_fill="#FFFFFF",
    color_accent="#1D1F21", color_accent_soft="#FBEFB5",
    button_primary_background_fill="#1D1F21", button_primary_background_fill_hover="#383C40",
    button_primary_text_color="#FFFFFF", button_primary_border_color="#1D1F21",
    button_large_text_size="1rem", button_large_padding="14px 20px",
    link_text_color="#1D1F21",
)
CSS = """
:root { color-scheme: light; --ink:#1D1F21; --graphite:#5A6068; --rule:#E2E5E8; --shade:#F5F6F7; --marker:#F7D046;
        --alarm:#C8372D; --alarm-hl:#F8D2CD; --clear:#1E7F4F; --clear-hl:#CFEBD9; }
html, body { background: #FFFFFF !important; color: var(--ink) !important; }
.gradio-container { max-width: 1040px !important; width: 100% !important; margin: 0 auto !important; }
.gradio-container .main, .gradio-container .contain, .gradio-container .wrap.main { width: 100% !important; }
footer { display: none !important; }

/* Gradio's built-in upload prompt follows the browser language, not our switch: replace it. */
:root { --drop-text: "将文件拖到这里，或点击选择"; }
body[data-lang="English"] { --drop-text: "Drop a file here, or click to choose one"; }
body[data-lang="한국어"] { --drop-text: "파일을 여기에 놓거나 클릭해서 선택하세요"; }
[data-testid="upload-text"] { font-size: 0 !important; }
[data-testid="upload-text"] .or { display: none; }
[data-testid="upload-text"]::after { content: var(--drop-text); display: block; font-size: .98rem; color: var(--graphite);
  margin-top: 8px; }
:focus-visible { outline: 3px solid var(--marker) !important; outline-offset: 2px; }

/* top bar: name left, language right, one rule under it */
#masthead { align-items: center; gap: 16px; padding: 22px 0 18px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }
#brand h1 { font-size: 1.5rem; line-height: 1.2; font-weight: 700; margin: 0; color: var(--ink); }
#brand h1 .hl { background: linear-gradient(transparent 58%, var(--marker) 58%, var(--marker) 92%, transparent 92%);
                padding: 0 .08em; }
#brand p { margin: 4px 0 0; color: var(--graphite); font-size: .95rem; }
#lang, #lang fieldset, #lang > div { background: transparent !important; border: 0 !important; padding: 0 !important;
  box-shadow: none !important; }
.form:has(#lang), div:has(> #lang) { background: transparent !important; border: 0 !important; box-shadow: none !important; }
#lang > span, #lang .block-title { display: none; }
#lang .wrap { gap: 4px !important; width: fit-content; margin-left: auto; border: 0 !important; }
#lang label { margin: 0 !important; border: 0 !important; border-radius: 0 !important; padding: 4px 6px !important;
  background: transparent !important; box-shadow: none !important; cursor: pointer; color: var(--graphite); }
#lang input { position: absolute; opacity: 0; width: 1px; height: 1px; }
#lang label:has(input:checked) { color: var(--ink) !important; font-weight: 600;
  background: linear-gradient(transparent 62%, var(--marker) 62%) !important; }

/* tabs: ink text, highlighter underline */
button[role="tab"] { font-size: .98rem !important; color: var(--graphite) !important; }
button[role="tab"][aria-selected="true"] { color: var(--ink) !important; font-weight: 600; }
button[role="tab"][aria-selected="true"]::after, .tab-wrapper button.selected::after { background: var(--marker) !important; height: 3px !important; }

/* step 1: the drop stage, full width */
#stage { border: 1.5px dashed #C9CED3 !important; background: var(--shade) !important; }
#stage:hover { border-color: var(--ink) !important; }
#stage .wrap, #stage button { background: transparent !important; }
#run-row { margin: 4px 0 10px; }

/* step 2: verdict strip */
.report { display: grid; grid-template-columns: minmax(240px, 1fr) minmax(0, 1.6fr); column-gap: 32px; row-gap: 6px;
          align-items: start; border-top: 2px solid var(--ink); padding: 22px 2px 4px; }
.report .lead { min-width: 0; }
.report.single { grid-template-columns: 1fr; }
.report.single .evidence { display: none; }
.report .verdict { font-size: clamp(1.7rem, 3vw, 2.3rem); font-weight: 700; line-height: 1.15; color: var(--ink);
                   display: inline; padding: 0 .1em; }
.report.fake .verdict { background: linear-gradient(transparent 55%, var(--alarm-hl) 55%); }
.report.real .verdict { background: linear-gradient(transparent 55%, var(--clear-hl) 55%); }
.report .status { display: block; margin-bottom: 10px; font-size: .9rem; font-weight: 600; }
.report .status::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                          margin-right: 8px; background: var(--graphite); vertical-align: 0; }
.report.fake .status { color: var(--alarm); } .report.fake .status::before { background: var(--alarm); }
.report.real .status { color: var(--clear); } .report.real .status::before { background: var(--clear); }
.report .detail { margin: 12px 0 0; color: var(--ink); font-size: 1rem; }
.report .airate { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; margin: 4px 0 10px;
                  color: var(--ink); }
.report .airate.r0 { color: #1E7F4F; } .report .airate.r1 { color: #5E8A1F; } .report .airate.r2 { color: #B07A00; }
.report .airate.r3 { color: #D2561E; } .report .airate.r4 { color: #C8372D; }
.report .airate b { font-size: clamp(2.8rem, 6vw, 3.8rem); font-weight: 800; line-height: 1;
                    font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.report .airate-label { font-size: 1.15rem; font-weight: 700; }
.report .airate b, .report .airate-label { color: inherit !important; }  /* Gradio prose colours <b> */
.report .airate small { flex-basis: 100%; color: var(--graphite); font-size: .8rem; }
.report .detail code { background: var(--shade); padding: 1px 6px; border-radius: 3px; font-size: .85em; }
.report .evidence { min-width: 0; overflow-x: auto; }
.report .note { grid-column: 1 / -1; margin: 14px 0 0; color: var(--graphite); font-size: .86rem; }
.report.empty.single, .report.error { display: block; border-top: 1px solid var(--rule); color: var(--graphite); }
.report.empty .note { margin: 0; font-size: 1rem; }
.report.error .verdict { font-size: 1.3rem; }
.report.loading .verdict { font-size: 1.3rem; }
.report.loading .verdict::after { content: "…"; }
.meter.progress { display: block; height: 4px; background: var(--rule); }
.meter.progress i { background: var(--ink); transition: width .25s ease-out; }
.report tr[data-new] { animation: row-in .35s ease-out; }
.report tr[data-new] .meter i { transform-origin: left; animation: bar-in .5s ease-out; }
@keyframes row-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes bar-in { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.meter { position: relative; height: 10px; background: #E9ECEF; margin: 14px 0 2px; max-width: 460px; }
.meter i { position: absolute; inset: 0 auto 0 0; background: var(--clear); }
.meter.fake i { background: var(--alarm); }
.meter b { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--ink); }
.report table.dets { width: 100%; border-collapse: collapse !important; font-size: .94rem; border: 0 !important;
  background: transparent !important; margin: 0; }
.report table.dets tr { border: 0 !important; background: transparent !important; }
.report table.dets th, .report table.dets td { border: 0 !important; border-bottom: 1px solid var(--rule) !important;
  background: transparent !important; white-space: nowrap; }
.report table.dets td:first-child { white-space: normal; }
.report table.dets th { text-align: left; font-weight: 500; color: var(--graphite); padding: 4px 8px 8px; }
.report table.dets td { padding: 12px 8px; vertical-align: middle; }
.report table.dets td .meter { margin: 0 10px 0 0; display: inline-block; width: 120px; vertical-align: middle; }
.report table.dets .score { display: inline-block; width: 40px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
.report table.dets tr.fake .score { color: var(--alarm); }
.report table.dets tr.real .score { color: var(--clear); }
.report table.dets .num { text-align: right; font-variant-numeric: tabular-nums; color: var(--graphite); }

/* step 3: evidence (heatmap / frames) */
.heat, #frames { border-color: var(--rule) !important; }
@media (max-width: 760px) {
  #masthead { flex-direction: column; align-items: flex-start; }
  #lang .wrap { margin-left: 0; }
  .report { grid-template-columns: 1fr; }
  .report table.dets .num { display: table-cell; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""
# The design is light-only: drop Gradio's automatic dark mode (it follows the OS setting).
FORCE_LIGHT = """() => {
  const light = () => document.querySelectorAll('.dark').forEach((el) => el.classList.remove('dark'));
  light();
  new MutationObserver(light).observe(document.documentElement, { subtree: true, attributes: true, attributeFilter: ['class'] });
}"""
STYLE = dict(theme=THEME, css=CSS, js=FORCE_LIGHT)


def brand(lang="zh"):
    t = TXT[lang]
    head = t["title"].split("\n", 1)[0].lstrip("# ")
    return f'<div id="brand"><h1><span class="hl">{esc(head)}</span></h1><p>{esc(t["sub"])}</p></div>'


with gr.Blocks(title="AI Face Detector", delete_cache=(3600, 3600)) as demo:
    with gr.Row(elem_id="masthead"):
        title = gr.HTML(brand("zh"), scale=3)
        lang = gr.Radio(list(LANGS), value="中文", label="Language / 언어 / 语言", show_label=False,
                        elem_id="lang", scale=1, min_width=220)
    with gr.Tab(TXT["zh"]["tab_auto"]) as tab_auto:
        up = gr.File(label=TXT["zh"]["upload"], type="filepath", file_types=["image", "video", ".heic", ".heif"],
                     elem_id="stage", height=230)
        with gr.Row(elem_id="run-row"):
            go = gr.Button(TXT["zh"]["detect"], variant="primary", size="lg")
        umd = gr.HTML(report("empty", "", "", t=TXT["zh"], note=TXT["zh"]["empty"]))
        ulbl = gr.Label(label=TXT["zh"]["prob"], visible=False)
        ucam = gr.Gallery(label=CAM_TEXT["zh"]["title"], columns=3, height=650,
                           object_fit="contain", visible=False, elem_id="model-heatmaps")
        ugal = gr.Gallery(label=TXT["zh"]["gallery"], columns=6, rows=2, height=360, object_fit="cover",
                          visible=False, elem_id="frames")
        go.click(analyze_any_stream, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any",
                 concurrency_id="detector", concurrency_limit=1, show_progress="hidden")
        up.upload(analyze_any_stream, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any_upload",
                  concurrency_id="detector", concurrency_limit=1, show_progress="hidden")
    with gr.Tab(TXT["zh"]["tab_pick"]) as tab_pick:
        with gr.Row(equal_height=True):
            inp = gr.Image(type="pil", label=TXT["zh"]["img_in"], height=300, scale=3)
            with gr.Column(scale=2):
                run = gr.Dropdown(run_names(), label=TXT["zh"]["model"], allow_custom_value=False)
                btn = gr.Button(TXT["zh"]["detect"], variant="primary", size="lg")
        md = gr.HTML(report("empty", "", "", t=TXT["zh"], note=TXT["zh"]["empty"]))
        lbl = gr.Label(label=TXT["zh"]["prob"], visible=False)
        cam = gr.Image(label=TXT["zh"]["cam_pick"], height=380, visible=False, elem_classes=["heat"])
        btn.click(analyze_pick_ui, [inp, run, lang], [lbl, cam, md], concurrency_id="detector", concurrency_limit=1)
        inp.change(analyze_pick_ui, [inp, run, lang], [lbl, cam, md], concurrency_id="detector", concurrency_limit=1)
    with gr.Tab(TXT["zh"]["tab_arms"]) as tab_arms:
        arms_intro = gr.Markdown(TXT["zh"]["arms_intro"])
        table = gr.Markdown()

    def relabel(lang_name):
        code = LANGS[lang_name]
        t = TXT[code]
        empty = report("empty", "", "", t=t, note=t["empty"])
        # Old results are cleared so they do not stay in the previous language.
        return (gr.update(value=brand(code)), gr.update(label=t["tab_auto"]), gr.update(label=t["tab_pick"]),
                gr.update(label=t["tab_arms"]), gr.update(label=t["upload"]), gr.update(value=t["detect"]),
                gr.update(label=t["prob"], value=None), gr.update(label=CAM_TEXT[code]["title"], value=None, visible=False),
                gr.update(label=t["gallery"], value=None, visible=False), gr.update(label=t["img_in"]),
                gr.update(label=t["model"]), gr.update(value=t["detect"]),
                gr.update(label=t["prob"], value=None), gr.update(label=t["cam_pick"], value=None, visible=False),
                gr.update(value=t["arms_intro"]), gr.update(value=results_table(code)),
                gr.update(value=empty), gr.update(value=empty))

    lang.change(relabel, lang, [title, tab_auto, tab_pick, tab_arms, up, go, ulbl, ucam, ugal,
                                inp, run, btn, lbl, cam, arms_intro, table, umd, md],
                js="(v) => { document.body.dataset.lang = v; return v; }")

    def refresh():
        start_preload()
        names = run_names()
        value = image_model()
        return gr.update(choices=names, value=value), results_table("zh")

    demo.load(refresh, None, [run, table])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    start_preload()
    demo.launch(**STYLE)
