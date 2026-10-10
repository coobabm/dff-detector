[中文](README.md) | [English](README.en.md) | **한국어**

# AI 생성 얼굴 판별: DeepFakeFace 재현 + OpenAI 이미지 + AI 영상 / 얼굴 교체 탐지

M2 Max Mac 한 대에서 "실제 vs AI 생성 얼굴" 탐지기를 처음부터 학습시키고, 실험으로 세 가지 질문에 답합니다.

1. **탐지기는 학습 때 보지 못한 생성기를 알아볼 수 있을까?** (탐지기와 생성기의 *군비 경쟁(arms race)* 재현)
2. **새 생성기를 배우려면 그 생성기의 이미지가 몇 장 필요할까?** (OpenAI gpt-image 계열로 실험)
3. **영상도 탐지할 수 있을까?** (FaceFusion 얼굴 교체, Veo / Sora / Kling 등 AI 생성 영상)

Gradio 웹 UI 포함 (中文 / English / 한국어): 이미지와 영상을 한 업로드 칸에 올리면 알맞은 전용 모델로 자동 전달됩니다. 이미지는 AI 생성 확률 + Grad-CAM 히트맵, 영상은 "AI 생성 영상"과 "얼굴 교체" 탐지기를 동시에 실행합니다.

![OpenAI 생성 예시](docs/openai_samples.jpg)
*위 얼굴은 모두 gpt-image-1.5 / 2 / 2.5로 생성했습니다. 실존 인물은 한 명도 없습니다.*

## 목차

- [주요 결과](#주요-결과)
- [영상 탐지](#영상-탐지)
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
| StyleGAN 얼굴 (처음 봄) | 탐지율 0% → 4,500장 추가 후 **87–99%** |
| FaceFusion 얼굴 교체 영상 | 넓은 크롭 + 작은 모델: 못 잡음 → **얼굴만 크롭 + EfficientNet-B4: AUC 0.93–0.98, 로컬 얼굴 교체 테스트 영상 4개 모두 탐지** |
| AI 생성 영상 (Veo / Sora / Kling / 디지털 휴먼) | 테스트 AUC **0.968**, 실제 영상 93.8% 정확; 단 **온라인에 재업로드·재압축된 영상은 아직 불안정** |

**한 줄 요약: 탐지기는 학습 때 본 종류의 생성기만 알아본다. 하지만 새 생성기 샘플 몇백 개면 금방 따라잡는다. 영상의 핵심은 "얼굴을 크게 보는 것".**

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

### 3. StyleGAN (GAN 생성 얼굴)

이미지 모델은 확산 모델(SD, OpenAI)만 봤기 때문에 StyleGAN 얼굴을 0% 탐지했습니다. StyleGAN2/3 얼굴 4,500장과 FFHQ 실제 얼굴(StyleGAN은 FFHQ로 학습되었으므로 가장 어려운 반례)을 추가해 최종 이미지 모델 `all_image`를 만들었습니다:

| 테스트 세트 (학습에 미사용) | 추가 전 | 추가 후 |
|---|---|---|
| thispersondoesnotexist (StyleGAN2) | 0% | 86.8% |
| StyleGAN2-Face | 0.6% | 99.4% |
| StyleGAN3 | 0% | 97.3% |
| 140k Real/Fake (StyleGAN) | 0% | 79.4% |
| OpenAI 이미지 | 98.5% | 98.5% |
| 실제 사진 (6개 출처) | 97.8% | 97.0% (FFHQ 96.4%) |
| 기존 DFF 가짜 | 79.4% | 77.5% |

## 영상 탐지

영상은 프레임 단위로 판별합니다: 프레임 추출 → YuNet으로 가장 큰 얼굴 검출 → 점수 → 전체 프레임의 중앙값.

### 실패한 방법: 넓은 크롭 + 작은 모델 (4라운드)

처음에는 이미지 모델과 같은 구도(얼굴이 화면의 약 45%, 224픽셀로 줄이면 얼굴 폭이 약 100픽셀)로 EfficientNet-B0를 영상 프레임에 미세조정했습니다:

| 라운드 | 추가한 것 | 결과 |
|---|---|---|
| 1 | 공개 AI 영상 / 얼굴 교체 데이터셋의 프레임, 실제 영상은 FF++만 | 거의 모든 영상을 "가짜"로 판정: 실제 영상 59%만 정확 |
| 2 | CelebV-HQ 실제 영상 추가 | 실제 영상 88% 정확, 하지만 로컬 AI 영상을 다시 모두 놓침 |
| 3 | 이 Mac의 FaceFusion으로 얼굴 교체 영상 약 300개 생성 (OpenAI 생성 얼굴을 교체) | 테스트 AUC 0.80–0.92, 하지만 로컬 아이언맨 교체 영상은 여전히 0점 |
| 4 | 실제 인물(FFHQ) 얼굴을 영화 장면에 교체, "영화 느낌" 증강 추가 | 새 교체 영상 AUC 0.67–0.77; 아이언맨 여전히 놓침 |

결론: 고품질 얼굴 교체의 흔적(경계, 피부 질감, 눈과 치아)은 매우 미세해서 100픽셀 얼굴에서는 거의 보이지 않습니다. 3라운드의 좋은 점수는 "교체된 얼굴 자체가 AI 생성"이라는 지름길 덕분이기도 했습니다.

### 효과가 있던 방법: 얼굴만 380픽셀 크롭 + EfficientNet-B4

얼굴 교체 탐지 연구의 표준 방식으로 바꿨습니다: 얼굴 박스의 1.3배만 잘라 380×380으로 확대(`extract_tight.py`), 더 큰 EfficientNet-B4(`train_swap.py`), 영상마다 16프레임. 실제와 가짜 영상을 쌍으로 만들어 같은 화질로 인코딩하고, 임계값은 검증 세트에서 고릅니다.

**얼굴 교체 탐지기** `runs_swap/b4` (영상 1,014개: 실제 421 / 교체 593):

| 테스트 (학습에 없던 인물 / 영상) | 결과 |
|---|---|
| FaceFusion hyperswap 1a/1b/1c, alphaface, inswapper | AUC 0.93–0.98, 탐지 75–100% |
| DF40 inswap / simswap | AUC 0.97–0.98, 탐지 100% |
| FF++ (Deepfakes, FaceSwap 등 5종) | AUC 0.80–0.96 |
| 실제 영상 | 90% 정확 |
| **로컬 아이언맨 얼굴 교체 영상 ×4** (작성자 본인 사진으로 교체, 학습에 전혀 미사용) | **4/4 탐지 (중앙값 1.00)**; 원본 0.00 |
| 원본 재인코딩 (q95/80/60, 720p) | 모두 진짜로 판정: 모델은 "재압축"이 아니라 얼굴 교체를 봄 |
| 교체 영상을 720p / crf 28로 재압축 | 여전히 가짜로 판정 (0.98) |

**AI 생성 영상 탐지기** `runs_aivid/b4` (Veo 2/3/3.1, Sora 1/2, Kling, Wan, Pika, Ray2, Hunyuan, 디지털 휴먼 / 말하는 얼굴 6종, HeyGen; 가짜 2,519개 + 실제 2,584개, "온라인 재압축" 사본 포함):

| 버전 | 변경 | 테스트 AUC | 실제 영상 정확도 | 가짜 탐지율 |
|---|---|---|---|---|
| v1 | 초판 | 0.80 | 87% | 57% |
| v2 | + TalkingHeadBench + CelebV 실제 영상 | 0.96 | 84% | 92% |
| v3 | + 모든 영상의 YouTube식 재압축 사본 (360–720p, H.264 / VP9) | 0.96 | 84% | 94% |
| **v4 (채택)** | 실제 영상 2배 (+CelebV 600개) | **0.968** | **93.8%** | 88% |

(네 버전 모두 같은 테스트 세트에서 비교: 재압축 사본 포함 991개 영상. 네 모델 앙상블도 v4보다 낫지 않았습니다.)

**미해결**: 온라인에 재업로드된 AI 영상 두 개(Wikimedia가 변환한 Veo 3 클립, YouTube의 AI 가짜 뉴스)는 네 버전 모두에서 점수가 임계값 근처를 오갑니다. 재압축 후 남은 흔적이 너무 적습니다. 또 실제 장면이 섞인 편집 영상은 실제 프레임 때문에 중앙값 점수가 낮아집니다.

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

### D. StyleGAN 얼굴 (`fetch_gan.py`)

| 출처 | 생성기 | 사용량 |
|---|---|---|
| [javi22/this-person-does-not-exist-10k](https://huggingface.co/datasets/javi22/this-person-does-not-exist-10k) | StyleGAN2 | 1,500 |
| [gojay/StyleGAN2-Face](https://huggingface.co/datasets/gojay/StyleGAN2-Face) | StyleGAN2 | 1,000 |
| [34data/StyleGAN3_1024x1024_images](https://huggingface.co/datasets/34data/StyleGAN3_1024x1024_images) | StyleGAN3 | 1,000 |
| [TheKernel01/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) | StyleGAN 가짜 + FFHQ 실제 | 각 1,000 |

출처마다 무작위 20%를 테스트용으로 분리.

### E. 영상

| 종류 | 출처 | 방법 |
|---|---|---|
| 얼굴 교체 (직접 생성) | 이 Mac의 FaceFusion 3.9: hyperswap 1a/1b/1c, alphaface, inswapper; FFHQ 실제 얼굴을 CelebV-HQ 인터뷰 클립과 원본 YouTube 영상(영화·드라마 포함)의 4초 구간에 교체 | `gen_faceswap.py --targets clips/raw --faces ffhq --keep`; 원본과 교체본을 같은 화질로 한 번씩 인코딩 |
| 얼굴 교체 (공개) | [FaceForensics++ C23](https://huggingface.co/datasets/bitmind/FaceForensicsC23), [DF40](https://huggingface.co/datasets/ohjoonhee/DF40-Fake-Videos) | `fetch_public_videos.py`; zip에서 필요한 영상만 받음 (HTTP range) |
| AI 생성 영상 | [34data gen-videos](https://huggingface.co/datasets/34data/gen-videos-veo3) (Veo 3 / Sora 2 / Kling / Wan 2), [GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench) (Sora / Kling), [Rapidata](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo3) 선호도 데이터셋 (Veo 2/3/3.1, Sora, Pika, Ray2, Hunyuan, Wan, Alpha) | `fetch_rapidata.py`, `fetch_aivid_videos.py`; 얼굴이 검출되는 영상만 보관 |
| 디지털 휴먼 / 말하는 얼굴 | [TalkingHeadBench](https://huggingface.co/datasets/luchaoqi/TalkingHeadBench) (LivePortrait, Hallo 1/2, AniPortrait ×2, EmoPortrait), DF40 HeyGen | 위와 같음 |
| 실제 영상 | FF++ 원본, [CelebV-HQ](https://huggingface.co/datasets/mengtingwei/celebv-HQ-raw-videos) 원본 영상 구간과 [인터뷰 클립](https://huggingface.co/datasets/saeed-5959/celebv_hq_head_talking) | 위와 같음 |
| 재압축 사본 | 위 영상마다 하나: 360/480/720p, H.264 crf 26–34 또는 VP9 | `recompress_videos.py` |

분할은 인물 / 프롬프트 단위로 묶습니다 (실제 클립과 그 교체본은 같은 그룹). 테스트 세트의 인물과 영상은 학습에 나오지 않습니다. 영상과 추출한 얼굴 이미지는 배포하지 않습니다.

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
| StyleGAN | `gan_finetune.py` | `all_openai`에서 4 에폭 미세조정. 실제: 사진 6개 출처 + 테스트 외 모든 FFHQ + 140k 실제; 가짜: StyleGAN 4개 출처 균등 + DFF + OpenAI → `runs/all_image` |
| 영상, 넓은 크롭 (구) | `video_scan.py`, `fetch_video.py`, `video_finetune.py` | 위 "실패한 방법" |
| 영상 얼굴 크롭 | `extract_tight.py` | 영상마다 16프레임, 얼굴 박스 1.3배, 380×380 |
| B4 탐지기 | `train_swap.py` | EfficientNet-B4, 배치 16, lr 1e-4, 5 에폭 × 1,500 스텝; 클래스·출처별 균형 샘플링; 학습 그룹 10%로 최적 에폭과 임계값 선택; `--task swap` 또는 `--task aivid`; "영화 느낌" 증강(어둡게, 색보정, 그레인) |
| 얼굴 교체 생성 | `gen_faceswap.py` | 로컬 FaceFusion 3.9 headless 실행; 모델 5개 순환, pixel boost 256/512, mask = box + occlusion (xseg_3) |
| 영상 스캔 | `scan_swap.py` | `--model runs_swap/b4` 또는 `runs_aivid/b4`, 32프레임, 접촉 시트 |
| 웹 UI | `app.py` | 업로드 칸 하나: 이미지 → `all_image` + Grad-CAM; 영상 → B4 탐지기 2개, 하나라도 임계값을 넘으면 의심으로 판정하고 가장 가까운 유형 표시; UI 3개 언어 |

## 빠른 시작 웹 UI

[uv](https://docs.astral.sh/uv/)가 필요합니다. Apple Silicon에서는 MPS, NVIDIA에서는 CUDA, 그 외에는 CPU를 사용합니다.

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
# 모델 가중치를 받아 저장소 루트에서 압축 해제 (runs*/…/model.pt 생성)
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.2/models.zip
unzip models.zip
uv run app.py      # http://127.0.0.1:7860 접속 후 "탐지" 탭에서 이미지나 영상 업로드
```

`models.zip` (v0.2) 구성:

| 폴더 | 용도 |
|---|---|
| `runs/all_image/` | **이미지**: DFF 세 종류 + OpenAI + StyleGAN, 웹 UI 기본값 |
| `runs_swap/b4/` | **영상 얼굴 교체** 탐지기 (EfficientNet-B4, 임계값 포함) |
| `runs_aivid/b4/` | **AI 생성 영상** 탐지기 (EfficientNet-B4 v4, 임계값 포함) |
| `runs/all_openai/`, `runs/all/`, `runs/t2i/` `runs/inpaint/` `runs/insight/`, `runs/all_video/` | 비교 기준 / 군비 경쟁 실험 / 구 넓은 크롭 영상 모델 |

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

# 8. StyleGAN (약 1시간) → runs/all_image
uv run fetch_gan.py
uv run gan_finetune.py

# 9. 영상 데이터 (약 15GB 다운로드; 얼굴 교체 생성은 M2 Max에서 약 5시간, 로컬 FaceFusion 3.x 필요)
uv run gen_faceswap.py --clips 120 --targets raw   --faces ffhq --tag hq_raw  --keep
uv run gen_faceswap.py --clips 120 --targets clips --faces ffhq --tag hq_clip --keep
uv run fetch_public_videos.py
uv run fetch_rapidata.py
uv run fetch_aivid_videos.py            # zips thb celebv celebvclips celebv2
uv run recompress_videos.py aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv

# 10. 얼굴 교체 탐지기 (약 2시간)
uv run extract_tight.py --lists hq_raw.csv hq_clip.csv public.csv --out swap_frames.csv
uv run train_swap.py

# 11. AI 생성 영상 탐지기 (약 2.5시간)
uv run extract_tight.py --lists aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv recomp.csv --out aivid_frames.csv
uv run train_swap.py --frames aivid_frames.csv --task aivid --out runs_aivid/b4

# 12. 영상 스캔
uv run scan_swap.py your_video.mp4                       # 얼굴 교체
uv run scan_swap.py --model runs_aivid/b4 your_video.mp4 # AI 생성
```

참고:

- 무작위 시드는 모두 0으로 고정했지만 GPU 학습은 완전히 결정적이지 않아, 재현 수치가 조금 다를 수 있음 (보통 ±1%).
- Pexels 이미지는 Pexels CDN에서 받으며, 작성자가 삭제한 사진은 자동으로 건너뜀.
- `gen_openai.py`는 이미 있는 이미지를 건너뛰므로, 중단되면 다시 실행해 이어서 생성 가능.

## 한계

- OpenAI 테스트 이미지와 학습 이미지가 같은 프롬프트 템플릿에서 나옴. 셀카, 단체 사진, 메신저 앱으로 재압축된 이미지는 점수가 더 낮을 가능성이 큼.
- 얼굴만 다룸. 이미지: SD 1.5, InsightFace, gpt-image, StyleGAN2/3; Midjourney, Flux, Gemini 등은 테스트하지 않음.
- 영상: 프레임 단위 모델이라 시간 정보(깜빡임, 입 모양과 소리의 일치)를 쓰지 않음. 온라인에서 재업로드·재압축된 AI 영상은 아직 불안정; 실제 영상의 약 6%가 여전히 오탐됨.
- 얼굴 교체 탐지기를 테스트한 FaceFusion 영상은 학습 데이터와 같은 도구에서 나옴 (인물·장면·교체 얼굴은 다름). DeepFaceLab 등 다른 도구는 FF++ / DF40으로만 테스트함.
- 새 생성기가 나오면 새 샘플로 다시 미세조정해야 함. 이것이 바로 군비 경쟁.

## 라이선스

- 코드: MIT ([LICENSE](LICENSE) 참고).
- OpenAI 생성 이미지 (릴리스의 `openai_faces_*` 및 `docs/openai_samples.jpg`): 이 프로젝트에서 OpenAI API로 생성, CC BY 4.0으로 공유. 출처를 밝혀 주세요. 이미지 속 인물은 모두 실존하지 않음.
- 제3자 데이터셋: 이 저장소에서 **재배포하지 않으며** 파일 목록만 제공. 각 라이선스 적용: DFF (OpenRAIL), FFHQ (CC BY-NC-SA 4.0), EasyPortrait (CC BY-SA 4.0), Pexels (Pexels License), Flickr30k / PPR10k (연구 목적 한정). 영상 데이터셋(FF++, DF40, CelebV-HQ, TalkingHeadBench, GenVidBench, Rapidata, 34data, StyleGAN 각 세트)도 로컬에서만 사용했으니 각 약관을 따르세요. FaceForensics++ 공식 버전은 이용 약관 동의가 필요합니다.

연구 및 교육 목적 전용입니다. 탐지 결과는 진위 판정의 근거가 될 수 없습니다.
