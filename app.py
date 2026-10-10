"""Web UI for AI-face detection (中文 / English / 한국어).

  uv run app.py   ->  http://127.0.0.1:7860

"Detect" tab: one upload box for images and videos, routed to specialist models:
  image -> all_image (AI images incl. StyleGAN; falls back to all_openai) + Grad-CAM
  video -> AI-video detector runs_aivid/b4 + face-swap detector runs_swap/b4
           (both EfficientNet-B4 on tight 380 px face crops; all_video as fallback)
  Fake if any detector is above its threshold; the breakdown says which kind.
"Pick a model" tab: run any trained image model by hand.
"""
import html
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
    """Verdict strip. kind: fake | real | empty | error. Left: status + verdict; right: per-detector evidence."""
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
                 f'<th class="num">{esc(t["c_thr"])}</th><th class="num">{esc(t["c_frames"])}</th></tr></thead><tbody>')
        for name, p, thr, n in rows:
            body += (f'<tr class="{"fake" if p > thr else "real"}"><td>{esc(name)}</td>'
                     f'<td>{meter(p, thr)}<span class="score">{p:.2f}</span></td>'
                     f'<td class="num">{thr:.2f}</td><td class="num">{n}</td></tr>')
        body += "</tbody></table>"
    body += "</div>"
    if note:
        body += f'<p class="note">{esc(note)}</p>'
    layout = "" if rows else " single"
    return f'<section class="report {kind}{layout}" aria-live="polite">{body}</section>'


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
        return None, None, report("empty", "", "", t=t, note=t["empty"])
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
    detail = f'{esc(t["p_ai"])} <strong>{p:.1%}</strong> · <code>{esc(run)}</code>' + meter(p, 0.5)
    md = report("fake" if p > 0.5 else "real", t["v_fake_img"] if p > 0.5 else t["v_real_img"], detail,
                t=t, note=t["note"])
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
        head, _, rest = t["no_face"].partition("\n\n")
        return None, None, report("error", bare(head), esc(rest), t=t), []
    flagged = [r for r in rows if r[1] > r[2]]
    if flagged:
        head, _, kind_line = t["v_fake"].partition("\n\n")
        kind = max(flagged, key=lambda r: r[1] - r[2])[0]
        verdict, detail = bare(head), esc(kind_line.replace("**{k}**", "")) + f"<strong>{esc(kind)}</strong>"
    else:
        verdict, detail = bare(t["v_real"]), ""
    note = t["foot"] + ("" if swap else " " + t["no_swap"])
    top = max(rows, key=lambda r: r[1] - r[2])
    md = report("fake" if flagged else "real", verdict, detail, rows, t, note)
    return {t["susp"]: top[1], t["ok"]: 1 - top[1]}, None, md, gallery


def analyze_any_ui(file, lang_name="中文"):
    """UI wrapper: show the heatmap for images and the face-frame strip for videos, hide empty panels."""
    lbl, cam, md, gal = analyze_any(file, lang_name)
    return lbl, gr.update(value=cam, visible=cam is not None), md, gr.update(value=gal or None, visible=bool(gal))


def analyze_pick_ui(img, run, lang_name="中文"):
    lbl, cam, md = analyze_pick(img, run, lang_name)
    return lbl, gr.update(value=cam, visible=cam is not None), md


def analyze_any(file, lang_name="中文"):
    lang = LANGS.get(lang_name, "zh")
    if file is None:
        return None, None, report("empty", "", "", t=TXT[lang], note=TXT[lang]["empty"]), []
    path = Path(file)
    if path.suffix.lower() in VIDEO_EXT:
        return analyze_video(str(path), lang)
    try:
        img = Image.open(path)
    except Exception:
        head, _, rest = TXT[lang]["bad_file"].partition("\n\n")
        return None, None, report("error", bare(head), esc(rest), t=TXT[lang]), []
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
footer { opacity: .5; }
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
.report .detail code { background: var(--shade); padding: 1px 6px; border-radius: 3px; font-size: .85em; }
.report .evidence { min-width: 0; }
.report .note { grid-column: 1 / -1; margin: 14px 0 0; color: var(--graphite); font-size: .86rem; }
.report.empty, .report.error { display: block; border-top: 1px solid var(--rule); color: var(--graphite); }
.report.empty .note { margin: 0; font-size: 1rem; }
.report.error .verdict { font-size: 1.3rem; }
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
  .report table.dets .num { display: none; }
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
        ucam = gr.Image(label=TXT["zh"]["cam"], height=340, visible=False, elem_classes=["heat"])
        ugal = gr.Gallery(label=TXT["zh"]["gallery"], columns=6, rows=2, height=360, object_fit="cover",
                          visible=False, elem_id="frames")
        go.click(analyze_any_ui, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any",
                 concurrency_id="detector", concurrency_limit=1)
        up.upload(analyze_any_ui, [up, lang], [ulbl, ucam, umd, ugal], api_name="analyze_any_upload",
                  concurrency_id="detector", concurrency_limit=1)
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
                gr.update(label=t["prob"], value=None), gr.update(label=t["cam"], value=None, visible=False),
                gr.update(label=t["gallery"], value=None, visible=False), gr.update(label=t["img_in"]),
                gr.update(label=t["model"]), gr.update(value=t["detect"]),
                gr.update(label=t["prob"], value=None), gr.update(label=t["cam_pick"], value=None, visible=False),
                gr.update(value=t["arms_intro"]), gr.update(value=results_table(code)),
                gr.update(value=empty), gr.update(value=empty))

    lang.change(relabel, lang, [title, tab_auto, tab_pick, tab_arms, up, go, ulbl, ucam, ugal,
                                inp, run, btn, lbl, cam, arms_intro, table, umd, md])

    def refresh():
        names = run_names()
        value = image_model()
        return gr.update(choices=names, value=value), results_table("zh")

    demo.load(refresh, None, [run, table])

if __name__ == "__main__":
    demo.launch(**STYLE)
