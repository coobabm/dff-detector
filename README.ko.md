[中文](README.md) | [English](README.en.md) | **한국어**

# AI 생성 얼굴 판별: CLIP + DINOv2 조합 모델

얼굴 이미지가 실제 사진인지, AI가 생성·조작한 것인지 판별합니다. 이 문서는 최신 학습 라운드만 다룹니다: 무엇을 했는지, 어떤 데이터를 썼는지, 보류(held-out) 테스트 세트 결과.

Gradio 웹 UI(中文 / English / 한국어)를 포함합니다. 이미지를 올리면 판정, 보정된 "AI 확률", Grad-CAM 히트맵을 보여 주고, 동영상은 별도의 동영상 검출기로 보냅니다.

<img src="docs/round_ui.jpg" width="640" alt="Web UI">

*웹 UI: OpenAI로 생성한 얼굴(이 프로젝트에서 생성)을 AI로 판정, AI 확률 >99%. 아래는 두 구성원 모델의 Grad-CAM.*

## 최종 모델

두 모델의 P(fake) 평균, 임계값 **0.522**:

| 구성원 | 백본 | 학습 방식 |
|---|---|---|
| `fm_ft_clip_l14_lora` | CLIP ViT-L/14 (OpenAI) | **전체 레이어 LoRA** (이번 라운드 신규) |
| `fm_ft_dinov2_b` | DINOv2-B (registers 포함) | 마지막 4개 블록 해제 (이전 라운드, 유지) |

임계값: 보류한 실제 사진의 절반에서 97번째 백분위수, 즉 실제 사진의 약 97%를 실제로 판정합니다. 아래 수치는 모두 나머지 절반과 다른 테스트 세트에서 계산했습니다.

## 이번 라운드에서 한 일

1. **전체 레이어 LoRA**: CLIP ViT-L/14의 모든 블록 `qkv`, `proj`, `fc1`, `fc2`에 LoRA(r=16, alpha=32, dropout 0.05)를 붙이고, 분류 헤드와 마지막 LayerNorm은 전부 학습합니다. 원래 가중치는 고정(학습 파라미터 6.3M). LoRA 학습률 2e-4, 헤드 1e-3, AdamW, 200스텝 워밍업 + 코사인 감쇠, 5 epoch. 학습 후 LoRA를 원래 가중치에 병합해 일반 timm 모델로 저장하므로 추론에 peft가 필요 없습니다.
2. **손실에서 실제 이미지 가중치 증가**: `BCEWithLogitsLoss(pos_weight=0.4)`, 즉 실제 이미지가 가짜보다 2.5배 무겁습니다. 실제 얼굴을 AI로 오판하는 비용이 커져 점수가 전반적으로 내려갑니다: 이전 임계값 0.715, 이번 0.522로 0.5에 가까워졌습니다.
3. **같은 내용의 가짜 이미지 확대** (B-Free 방식): 실제 얼굴을 생성 모델의 오토인코더에 한 번 통과시키거나 일부 영역을 인페인팅합니다. 내용이 원본과 같으므로 모델은 "누구인지, 어디서 찍었는지" 같은 지름길 대신 생성 흔적을 배워야 합니다. 이번 라운드 신규: FLUX.2, Qwen-Image VAE, SD1.5 / SDXL / FLUX VAE 추가, SD2 인페인팅 1,190장까지.
4. **새 생성기 데이터 추가**: OpenFake 샤드 9개, 얼굴 약 12,000장 추가 (gpt-image, nano-banana, Imagen 4, Midjourney, Seedream 등과 짝을 이루는 LAION / Pexels 실제 사진).
5. **카메라 원본 실제 사진**: 작성자 본인 앨범에서 편집하지 않았고 EXIF에 편집 소프트웨어가 없으며 카메라 모델이 있는 얼굴 사진을 실제 이미지로 학습했습니다(로컬에서만 사용, 공개하지 않음).
6. **테스트 시 증강**: 좌우 반전 평균을 시험했지만 이 조합에서는 이득이 없어 껐습니다(`ensemble.json`에 `"tta": true`로 켤 수 있음).

![content-matched fakes](docs/round_aligned.jpg)
*같은 내용 가짜: 위는 실제, 가운데는 가짜, 아래는 차이를 8배 확대. VAE 왕복은 아주 옅은 흔적만 남기고, 인페인팅은 한 영역만 바꿉니다.*

epoch마다 실제 15,000장 + 가짜 15,000장을 계열별로 균형 있게 뽑습니다(계열 안에서 출처별 같은 비율). 모두 256 px 느슨한 얼굴 크롭, JPEG q95. 증강(두 클래스 동일): 축소 후 확대, 가우시안 블러, JPEG 50–95, 영화풍 색보정, 랜덤 크롭, 좌우 반전, 약한 색 변화.

### 학습 데이터

| 실제 | 장수 | 가짜 | 장수 |
|---|---|---|---|
| FairFace (모든 연령 / 피부색) | 25,940 | SFHQ-T2I (FLUX, SDXL, DALL-E 3) | 10,961 |
| 실제 동영상 프레임 | 8,155 | 얼굴 교체 동영상 프레임 (FaceFusion 등) | 7,430 |
| 실제 사진 (FFHQ / Pexels / PPR10k / Flickr30k / EasyPortrait / DFF 실제) | 5,958 | OpenFake (gpt-image, nano-banana, Imagen, MJ, Seedream …) | 7,064 |
| OpenFake 실제 (LAION / Pexels) | 5,656 | AI 동영상 프레임 (Veo / Sora / Kling / 디지털 휴먼 …) | 6,804 |
| StyleGAN 데이터셋의 실제 사진 | 4,660 | 같은 내용 VAE 왕복 (SD1.5 / SDXL / FLUX / FLUX.2 / Qwen-Image) | 6,458 |
| Reddit 사진 게시판 + 카메라 원본 | 2,138 | DeepFakeFace (SD 텍스트→이미지 / 인페인팅 / InsightFace 교체) | 6,000 |
| 같은 내용 가짜의 원본 실제 사진 | 1,190 | Reddit AI 아트 게시판 | 5,957 |
| | | StyleGAN2/3 | 3,648 |
| | | 같은 내용 SD2 인페인팅 | 1,190 |
| | | OpenAI gpt-image | 897 |

![training loss](docs/round_loss.png)

## 결과

![results](docs/round_results.png)

보류 테스트 세트 (학습에 쓰지 않은 인물 / 게시물 / 동영상; Rapidata는 학습에 전혀 쓰지 않음):

| 실제: 실제로 판정 | | AI: 검출 | |
|---|---|---|---|
| 실제 사진 | 96.7% | OpenAI gpt-image | 100% |
| 실제 동영상 프레임 | 95.3% | SFHQ (FLUX / SDXL / DALL-E 3) | 99.9% |
| FairFace | 99.7% | StyleGAN | 99.4% |
| OpenFake 실제 | 97.8% | DFF SD 텍스트→이미지 / 인페인팅 / 교체 | 97.2% / 94.6% / 76.2% |
| Reddit 사진 게시판 | 97.7% | **Rapidata (최신 생성기, 학습 안 함)** | **96.4%** |
| StyleGAN 데이터셋 실제 | 100% | Reddit AI 아트 | 89.7% |
| 같은 내용 가짜의 원본 실제 | 97.0% | OpenFake | 84.1% |
| | | AI 동영상 프레임 | 79.3% |
| | | 얼굴 교체 동영상 프레임 | 63.9% |
| | | 같은 내용 인페인팅 | 59.0% |
| | | 같은 내용 VAE 왕복 | 41.6% |

![samples](docs/round_samples.jpg)
*보류 세트에서 무작위로 뽑은 이미지(골라낸 것이 아님). 숫자는 조합 점수이며 빨간색은 임계값 초과. 첫 줄은 학습에 전혀 쓰지 않은 생성기이고, 둘째 줄의 Midjourney 6 이미지는 놓쳤습니다.*

- 실제 세트 평균 97.7% 정답, 가짜 세트 평균 82.1% 검출.
- CLIP LoRA 단독은 평균 88.0%를 검출하지만 실제 세트 정답률이 96.9%(실제 동영상 프레임 93.4%)로 떨어집니다. DINOv2와 평균하면 오탐이 이전 수준으로 돌아옵니다.
- 같은 내용 VAE 가짜가 가장 어렵습니다: 원본과 내용이 같고 오토인코더의 미세한 흔적만 남습니다.

**AI 확률**: 조합 점수를 보류 데이터에서 등위 회귀로 보정합니다(진짜·AI 반반 사전확률, 테스트 세트마다 같은 가중치). 보류 절반에서 ECE 3.0% → 0.5%. 점수 0.3 ≈ 24%, 0.5 ≈ 64%, 0.9 ≈ 99%. 판정은 여전히 임계값 기준입니다.

![calibration](docs/round_calibration.png)
*왼쪽: 보류 세트에서 실제와 AI 이미지의 점수 분포(로그 축). 오른쪽: 점수에서 AI 확률로 가는 보정 곡선.*

전체 수치: `results/fm_combo.json`(조합 선택), `results/arms_race.json`(모든 모델 × 모든 테스트 세트).

## 사용법

[uv](https://docs.astral.sh/uv/)가 필요합니다. Apple Silicon은 MPS, NVIDIA는 CUDA, 그 외에는 CPU를 씁니다.

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
# 모델 가중치(v0.3, 1.7 GB)를 프로젝트 루트에서 풀면 runs*/…/model.pt가 생깁니다
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.3/models-v0.3.zip
unzip models-v0.3.zip
uv run app.py      # http://127.0.0.1:7860
```

웹 UI는 `runs/fm_ens/ensemble.json`에서 구성원을, `runs/<구성원>/model.pt`에서 가중치를 읽습니다. `models-v0.3.zip`에는 이번 라운드 조합 모델(`runs/fm_ens/`, `runs/fm_ft_clip_l14_lora/`, `runs/fm_ft_dinov2_b/`)과 동영상 검출기(`runs_aivid/v4/` AI 생성 동영상, `runs_swap/b4/` 얼굴 교체)가 들어 있습니다. 비상업 라이선스 데이터(OpenFake, FFHQ 등)로 학습했으므로 비상업 연구에만 쓰세요. UnivFD / NPR / B-Free는 포함되지 않으며, 없어도 웹 UI는 정상 동작합니다.

## 저장소 구조

| 폴더 | 내용 |
|---|---|
| `app.py` | 웹 UI(원클릭 검출, 모델별 단일 이미지, 생성기 교차 표) |
| `detector/` | 추론: 얼굴 크롭(`video_scan.py`, `extract_tight.py`), 동영상 스캔(`scan_swap.py`), 공개 검출기 래퍼(`external_detectors.py`) |
| `data_prep/` | 데이터 다운로드·생성: `fetch_*.py`, `gen_*.py`, `prepare.py`, `recompress_videos.py` |
| `training/` | 학습: `fm_detector.py`(이번 라운드 CLIP / DINOv2), `train.py`, `train_swap.py` 등 |
| `evaluation/` | 평가: `fm_compare.py`, `calibrate.py`, `arms_race.py`, `make_figures.py` 등 |
| `data_lists/` | 데이터 목록(파일 이름 / 샤드), 이미지 없음 |
| `tests/` | 단위 테스트: `uv run python -m unittest discover -s tests` |

모든 스크립트는 프로젝트 루트에서 실행합니다. 예: `uv run training/fm_detector.py …`.

## 이번 라운드 재현

이전 라운드 데이터(DFF, OpenAI 생성 이미지, 현대 실제 사진, StyleGAN, 동영상 프레임)는 `data_prep/prepare.py`, `data_prep/gen_openai.py`, `data_prep/fetch_real.py`, `data_prep/fetch_gan.py`, `detector/extract_tight.py`로 준비합니다(커밋 기록 참고). 이번 라운드 추가분:

```bash
# 새 생성기 / 실제 사진 (Hugging Face)
uv run data_prep/fetch_hf.py sfhq
uv run data_prep/fetch_hf.py openfake
uv run data_prep/fetch_hf.py openfake data_lists/openfake_shards2.txt openfake2   # 이번 라운드에 추가한 샤드 9개
uv run data_prep/fetch_hf.py fairface
uv run data_prep/fetch_hf.py rapidata                                        # 테스트 전용
uv run data_prep/fetch_hf.py manifest

# Reddit (rdt-cli와 rdt login 필요)
uv run data_prep/fetch_reddit.py list && uv run data_prep/fetch_reddit.py list_real
uv run data_prep/fetch_reddit.py download && uv run data_prep/fetch_reddit.py faces

# 같은 내용 가짜: 1라운드, 이어서 2라운드 (새 VAE + 인페인팅 추가)
uv run data_prep/gen_aligned.py
uv run data_prep/gen_aligned.py --round 2 --train 6000 --test 1000 --max-inpaint 1200 --only vae_flux2 vae_qwen vae_sd15 vae_sdxl vae_flux
uv run data_prep/gen_aligned.py --round 2 --train 6000 --test 1000 --max-inpaint 1200 --only inpaint

# 선택: 본인 앨범의 카메라 원본 얼굴 (macOS 사진 앱)
uvx osxphotos query --json --not-edited --only-photos > /tmp/myphotos_query.json
uv run data_prep/fetch_myphotos.py /tmp/myphotos_query.json

# 학습
FM_LORA=16 FM_REAL_WEIGHT=2.5 FM_TAG=_lora uv run training/fm_detector.py finetune clip_l14 5
uv run training/fm_detector.py finetune dinov2_b 4   # 서비스 중인 모델은 이전 라운드 것이며, 현재 데이터로 다시 학습하면 수치가 조금 다릅니다

# 평가, 조합 선택, 보정
uv run evaluation/fm_compare.py fm_ft_clip_l14_lora fm_ft_dinov2_b
uv run evaluation/calibrate.py
uv run evaluation/arms_race.py
uv run evaluation/make_figures.py   # 이 문서의 그림
```

## 한계

- 얼굴 영역만 보고 배경이나 전체 이미지는 판단하지 않습니다.
- 얼굴 교체 프레임(64%)과 같은 내용 VAE 가짜(42%)는 아직 약합니다. 동영상 얼굴 교체는 별도 동영상 검출기에 의존합니다.
- 스크린샷이나 여러 번 재압축된 이미지는 점수가 낮게 나옵니다.
- 새 생성기가 나오면 데이터를 보충해 다시 학습해야 합니다.

## 라이선스

- 코드: MIT ([LICENSE](LICENSE) 참고).
- 제3자 데이터셋은 로컬에서만 사용하며 이 저장소에서 **재배포하지 않습니다**: FairFace(CC BY 4.0), SFHQ-T2I(MIT), OpenFake(CC BY-NC 4.0), Rapidata(CDLA-Permissive-2.0), DFF(OpenRAIL), FFHQ(CC BY-NC-SA 4.0), Reddit 게시물(Reddit 약관) 등.
- `docs/round_*` 그림의 예시 이미지는 Rapidata(CDLA-Permissive-2.0), SFHQ-T2I(MIT), OpenFake(CC BY-NC 4.0), FairFace(CC BY 4.0), Pexels(Pexels License)와 이 프로젝트에서 생성한 OpenAI 이미지이며, 설명용으로만 씁니다.
- 비교에 쓴 공개 검출기(UnivFD, NPR, B-Free)는 이 저장소에 없습니다. 각 프로젝트에서 받아 해당 라이선스를 따르세요.

연구·교육용입니다. 검출 결과는 감정 근거가 될 수 없습니다.
