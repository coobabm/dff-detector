[中文](README.md) | [English](README.en.md) | **한국어**

# AI 생성 얼굴 판별: DeepFakeFace 재현 + OpenAI 이미지 탐지

M2 Max Mac 한 대에서 "실제 사진 vs AI 생성 얼굴" 탐지기를 처음부터 학습시키고, 실험으로 두 가지 질문에 답합니다.

1. **탐지기는 학습 때 보지 못한 생성기를 알아볼 수 있을까?** (탐지기와 생성기의 *군비 경쟁(arms race)* 재현)
2. **새 생성기를 배우려면 그 생성기의 이미지가 몇 장 필요할까?** (OpenAI gpt-image 계열로 실험)

Gradio 웹 UI 포함: 얼굴 이미지를 올리면 → AI 생성 확률 + Grad-CAM 히트맵.

![OpenAI 생성 예시](docs/openai_samples.jpg)
*위 얼굴은 모두 gpt-image-1.5 / 2 / 2.5로 생성했습니다. 실존 인물은 한 명도 없습니다.*

## 목차

- [주요 결과](#주요-결과)
- [데이터 출처](#데이터-출처)
- [방법](#방법)
- [빠른 시작 웹 UI](#빠른-시작-웹-ui)
- [전체 재현](#전체-재현)
- [한계](#한계)
- [라이선스](#라이선스)

## 주요 결과

| | 결과 |
|---|---|
| 한 가지 생성 방식으로 학습, 보지 못한 다른 방식으로 테스트 | 자주 찍기 수준으로 떨어짐 (AUC 0.3–0.7) |
| DFF 생성 방식 세 가지 모두 학습 (`all`) | 세 종류 모두 AUC > 0.9, 하지만 OpenAI 이미지는 AUC 0.44 |
| OpenAI 이미지 **50장**으로 미세조정 | OpenAI 탐지율 3% → **83.5%** |
| **500장** | **98.0%**, 더 늘려도 거의 그대로 (897장: 98.5%) |
| 최신 실제 사진 오탐률 | 0–1.4% (Pexels / PPR10k / FFHQ / Flickr30k / EasyPortrait) |
| AI 영상 (Veo 3, ModelScope, FaceFusion 얼굴 교체) | **대부분 놓침**: 학습 데이터에 이런 종류가 없음 |

**한 줄 요약: 탐지기는 학습 때 본 종류의 생성기만 알아본다. 하지만 새 생성기 샘플 몇백 장이면 금방 따라잡는다.**

### 1. 생성기 교차 테스트 (DeepFakeFace)

![cross generator](docs/cross_generator.png)

- SD 텍스트→이미지로만 학습 → 얼굴 교체 AUC 0.49 (찍기 수준).
- 얼굴 교체로만 학습 → 텍스트→이미지 AUC **0.305**, 찍기보다도 나쁨: 확산 모델 가짜를 실제 사진보다 더 "진짜"로 판단.
- 인페인팅으로 학습한 모델이 가장 잘 전이됨 (텍스트→이미지 0.965). 인페인팅은 국소적인 이음매를 배우게 만들고, 이런 흔적이 더 일반적이기 때문.
- 테스트 세트에는 학습에 한 번도 나오지 않은 인물만 포함 (IMDB-WIKI 인물 ID로 분할).

### 2. OpenAI 학습 곡선

![learning curve](docs/learning_curve.png)

학습에 쓰지 않은 동일한 OpenAI 이미지 200장 + 6개 출처의 실제 사진 3,000장으로 테스트:

| 추가한 OpenAI 이미지 | OpenAI AUC | OpenAI 탐지율 | 실제 사진 정확도 | 기존 DFF 가짜 탐지율 |
|---|---|---|---|---|
| 0 | 0.412 | 3.0% | 95.2% | 76.7% |
| 50 | 0.976 | 83.5% | 98.1% | 79.6% |
| 100 | 0.991 | 90.5% | 98.0% | 80.3% |
| 200 | 0.994 | 93.5% | 98.1% | 79.5% |
| 500 | 0.998 | 98.0% | 97.6% | 81.3% |
| 897 | 0.999 | 98.5% | 97.8% | 79.4% |

gpt-image-1.5 / 2 / 2.5 모두 약 98% 탐지. 실제 사진 중에서는 DFF의 오래된 wiki 사진만 오탐률이 높음 (약 10%, 작고 흐릿하기 때문). 전체 수치는 [`results/`](results/) 참고.

### 3. 영상

초당 2프레임 → YuNet 얼굴 검출 → 얼굴마다 점수. 가장 좋은 모델도 Veo 3, ModelScope, AI 가짜 뉴스 영상, FaceFusion 3.9 얼굴 교체 영상에서 중앙값 점수가 0.3 미만으로, 대부분 놓침. 이유: 영상 생성기나 최신 얼굴 교체 도구의 학습 데이터가 없고, H.264 압축이 미세한 흔적을 지움.

## 데이터 출처

세 종류의 데이터, 총 약 15만 장. 모두 256×256 JPEG (품질 95)로 변환.

### A. DeepFakeFace (DFF): 주 학습 세트

- 출처: [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) (Hugging Face, OpenRAIL 라이선스, zip 4개, 약 5GB)
- 구성:

| zip | 분류 | 수량 | 생성 방식 |
|---|---|---|---|
| `wiki.zip` | 진짜 | 30,000 | IMDB-WIKI 유명인 사진 |
| `text2img.zip` | 가짜 | 30,000 | Stable Diffusion v1.5 텍스트→이미지 |
| `inpainting.zip` | 가짜 | 30,000 | SD 인페인팅 (실제 사진 일부를 다시 그림) |
| `insight.zip` | 가짜 | 30,000 | InsightFace 얼굴 교체 |

- 네 zip 모두 구조가 같음: `<분류>/<00–99>/<인물ID>_<생년월일>_<촬영연도>.jpg`. 가짜 이미지는 원본 실제 사진과 파일명이 같음.
- 하위 폴더 번호 = 인물 ID의 끝 두 자리. 한 사람은 항상 한 폴더에만 있으므로 폴더 단위로 나누면 인물이 섞이지 않음: **00–79 학습, 80–89 검증, 90–99 테스트**.
- 실제 사진은 크기가 제각각. DFF 저자들이 가짜를 만들기 전에 512×512로 늘렸기 때문에, 이 프로젝트도 전부 256×256으로 늘려 진짜와 가짜의 기하 구조를 맞춤.
- 전체 목록: [`data_lists/dff_manifest.csv`](data_lists/dff_manifest.csv).

### B. OpenAI 생성 얼굴: 이 프로젝트에서 직접 생성

- OpenAI Images API (`/v1/images/generations`, 1024×1024, 기본 medium 품질)로 생성.
- 모델 3종: gpt-image-2 (434장), gpt-image-1.5 (332장), gpt-image-2.5 (331장), **총 1,097장** (1,100장 요청, 3장은 API가 HTTP 400으로 거부).
- 프롬프트는 무작위 조합 (시드 0 고정, 완전 재현 가능):

```
Realistic photograph of a {나이} {인종} {성별} {장면}, face clearly visible, upper body, {촬영 스타일}. No text, no watermark.
```

  - 나이 6종 (young … elderly) × 인종 6종 × 성별 2종 × 장면 8종 (레드카펫, 기자회견, 거리, 사무실, 스포츠 경기, TV 스튜디오 등) × 스타일 5종 (2008년 디지털카메라, 통신사 사진, 휴대폰 스냅, 프로필 사진, 오래된 스캔 사진).
  - 장면과 스타일은 IMDB-WIKI 유명인 뉴스 사진을 일부러 흉내 내서, 진짜와 가짜의 주제가 비슷하도록 함.
- 원본은 PNG (약 1024–1300px)이며 OpenAI가 넣은 C2PA 콘텐츠 자격 증명이 포함됨. 학습 시에는 정사각형으로 가운데를 자른 뒤 256×256으로 축소.
- 다운로드: [Releases v0.1](../../releases/tag/v0.1)

| 파일 | 내용 |
|---|---|
| `openai_faces_256.zip` (26MB) | 256×256 JPEG 1,097장 (학습에 쓴 버전) |
| `openai_faces_orig_part1.zip` / `part2.zip` (각 1.1GB) | 원본 PNG |
| `openai_meta.json` | 이미지별 모델과 프롬프트 |

### C. 최신 실제 인물 사진: "선명하면 가짜" 지름길 차단

DFF의 실제 사진은 대부분 오래되고 흐릿한 반면, OpenAI 이미지는 선명하고 현대적. DFF 실제 사진만 쓰면 모델이 "선명한 최신 사진 = 가짜"로 배울 수 있음. 그래서 실제 사진 세트 5개를 추가:

| 출처 | 내용 | 사용량 | 추출 방법 |
|---|---|---|---|
| [FFHQ](https://huggingface.co/datasets/bitmind/ffhq-256) | Flickr 고해상도 얼굴 (정렬 크롭) | 4,375 | `bitmind/ffhq-256`의 첫 번째 parquet 샤드 |
| [Pexels Portrait](https://huggingface.co/datasets/gaunernst/pexels-portrait) | 전문 인물 사진 | 3,000 | Pexels ID < 14,000,000 (약 2023년 이전 업로드, AI 이미지 회피)만, 무작위 추출 |
| [PPR10k](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k) | 보정 전 인물 촬영 원본 | 3,000 | 샘플별 `before.jpg`, 무작위 추출 |
| [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) | 일상 사진 | 1,458 | zip에서 무작위 3,000개; 그중 약 절반이 macOS `._` 메타데이터 파일이라 제외 |
| [EasyPortrait](https://huggingface.co/datasets/gofixyourself/EasyPortrait) | 웹캠 셀카 (화상회의 환경) | 3,000 | `data/test.zip`에서 무작위 추출 |

- 모든 무작위 추출은 시드 0 고정. 정사각형이 아닌 사진은 가운데를 자르고, 세로 사진은 얼굴이 남도록 위쪽부터 자름.
- 이 사진들에는 식별 가능한 실제 인물이 있고 일부 라이선스는 재배포를 금지하므로, **이 저장소에는 포함하지 않음**. 사용한 모든 이미지의 ID는 [`data_lists/`](data_lists/)에 있음.

## 방법

| 단계 | 파일 | 설명 |
|---|---|---|
| 데이터 | `prepare.py` | DFF 다운로드 → 256×256 → 인물 폴더 단위 분할, `data/manifest.csv` 작성 |
| 학습 | `train.py` | EfficientNet-B0 (timm, ImageNet 사전학습) 이진 분류기. AdamW + OneCycle, lr 3e-4, 배치 64, 4 에폭, 검증 AUC가 가장 좋은 에폭 저장. 진짜와 가짜 모두에 같은 무작위 축소 / 블러 / JPEG / 크롭 / 뒤집기 / 색 변환을 적용해 선명도로 꼼수를 못 쓰게 함. 에폭마다 실제 사진 전부 + 같은 수의 가짜를 무작위 추출 (클래스 균형) |
| 실험 | `run_all.sh` | 모델 4개: `t2i`, `inpaint`, `insight`, `all` |
| OpenAI 가짜 | `gen_openai.py` | 위 B 참고 |
| 최신 실제 사진 | `fetch_real.py` | 위 C 참고 |
| 제로샷 테스트 | `eval_openai.py` | DFF 모델로 OpenAI 이미지를 바로 테스트 |
| 학습 곡선 | `openai_curve.py` | OpenAI 이미지 1,097장 중 200장은 테스트용으로 따로 두고, 나머지 897장이 학습 풀. `all`에서 시작해 N장 (0/50/100/200/500/897)으로 3 에폭 미세조정 (lr 1e-4). 에폭마다: 실제 사진 약 6,000장 (6개 출처에서 각 최대 1,000장) + DFF 가짜 3,000장 + OpenAI 3,000장 (N장을 반복 추출). 테스트 실제 사진 = 6개 출처 각 500장, 학습과 겹치지 않음 |
| 영상 | `video_scan.py` | 초당 2프레임, YuNet 얼굴 검출, DFF식 크롭 (얼굴이 폭의 약 45%), 점수, 접촉 시트 출력 |
| 웹 UI | `app.py` | Gradio: 업로드 → 확률 + Grad-CAM; 모델 전환; 생성기 교차 결과표 |

## 빠른 시작 웹 UI

[uv](https://docs.astral.sh/uv/)가 필요합니다. Apple Silicon에서는 MPS, NVIDIA에서는 CUDA, 그 외에는 CPU를 사용합니다.

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
# 모델 가중치를 받아 저장소 루트에서 압축 해제 (runs/*/model.pt 생성)
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.1/models.zip
unzip models.zip
uv run app.py      # http://127.0.0.1:7860 접속 후 all_openai 선택
```

`models.zip`에는 모델 5개가 들어 있습니다:

| 폴더 | 학습 데이터 | 용도 |
|---|---|---|
| `runs/all_openai/` | DFF 세 종류 + OpenAI 897장 + 실제 사진 6개 출처 | **추천**, 종합 성능 최고 |
| `runs/all/` | DFF 세 종류 | 비교 기준 |
| `runs/t2i/` `runs/inpaint/` `runs/insight/` | DFF 한 종류씩 | 군비 경쟁 실험 |

## 전체 재현

순서대로 실행합니다. 시간은 M2 Max (64GB) 실측 기준이며, 디스크는 약 20GB 필요합니다.

```bash
# 0. 환경
uv sync

# 1. DFF: zip 4개 (5GB) 다운로드 → 축소 → 분할 (약 10분 + 다운로드 시간)
uv run prepare.py

# 2. DFF 모델 4개 (약 3시간)
./run_all.sh
#    결과: runs/<name>/results.json, 로그: logs/

# 3. OpenAI 이미지
#    방법 a: 공개된 이미지 사용 (추천, 결과 동일)
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.1/openai_faces_256.zip
mkdir -p data/img256 data/openai
unzip openai_faces_256.zip -d data/img256 && mv data/img256/meta.json data/openai/
#    방법 b: 직접 생성 (약 2시간, API 키 필요; OpenAI 출력은 무작위라 이미지가 달라짐)
export OPENAI_API_KEY=sk-...
#    export OPENAI_BASE_URL=...   # OpenAI 호환 프록시를 쓸 때 설정
uv run gen_openai.py --n 100 --models gpt-image-2            # 0–99번
uv run gen_openai.py --n 1100 --start 100 \
    --models gpt-image-1.5 gpt-image-2 gpt-image-2.5          # 100–1099번, 모델 3개 순환

# 4. 최신 실제 사진 (약 30분, 약 10GB 다운로드; 처리 후 원본 압축 파일은 자동 삭제)
uv run fetch_real.py ffhq pexels ppr10k flickr30k easyportrait

# 5. DFF 모델로 OpenAI 이미지 제로샷 테스트 (몇 분)
uv run eval_openai.py

# 6. 학습 곡선 (약 40분) → runs_openai/curve.json, runs_openai/n*.pt
uv run openai_curve.py

# 7. 최종 모델을 웹 UI에 넣기
mkdir -p runs/all_openai && cp runs_openai/n897.pt runs/all_openai/model.pt
uv run app.py

# 8. (선택) 영상 스캔
uv run video_scan.py your_video.mp4   # → scans/<영상이름>.jpg / .json
```

참고:

- 무작위 시드는 모두 0으로 고정했지만 GPU 학습은 완전히 결정적이지 않아, 재현 수치가 조금 다를 수 있음 (보통 ±1%).
- Pexels 이미지는 Pexels CDN에서 받으며, 작성자가 삭제한 사진은 자동으로 건너뜀.
- `gen_openai.py`는 이미 있는 이미지를 건너뛰므로, 중단되면 다시 실행해 이어서 생성 가능.

## 한계

- OpenAI 테스트 이미지와 학습 이미지가 같은 프롬프트 템플릿에서 나옴. 셀카, 단체 사진, 메신저 앱으로 재압축된 이미지는 점수가 더 낮을 가능성이 큼.
- 얼굴만, 그리고 SD 1.5, InsightFace, gpt-image 계열만 다룸. Midjourney, Flux, Gemini 등은 테스트하지 않음.
- 단일 프레임 모델이라 영상의 시간 정보를 쓰지 않음.
- 새 생성기가 나오면 새 샘플로 다시 미세조정해야 함. 이것이 바로 군비 경쟁.

## 라이선스

- 코드: MIT ([LICENSE](LICENSE) 참고).
- OpenAI 생성 이미지 (릴리스의 `openai_faces_*` 및 `docs/openai_samples.jpg`): 이 프로젝트에서 OpenAI API로 생성, CC BY 4.0으로 공유. 출처를 밝혀 주세요. 이미지 속 인물은 모두 실존하지 않음.
- 제3자 데이터셋: 이 저장소에서 **재배포하지 않으며** 파일 목록만 제공. 각 라이선스 적용: DFF (OpenRAIL), FFHQ (CC BY-NC-SA 4.0), EasyPortrait (CC BY-SA 4.0), Pexels (Pexels License), Flickr30k / PPR10k (연구 목적 한정).

연구 및 교육 목적 전용입니다. 탐지 결과는 진위 판정의 근거가 될 수 없습니다.
