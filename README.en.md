[中文](README.md) | **English** | [한국어](README.ko.md)

# AI-generated face detection: CLIP + DINOv2 ensemble

Tells whether a face image is a real photo or AI-generated / manipulated. This page covers only the latest training round: what was done, which data was used, and the results on held-out test sets.

Includes a Gradio web UI (中文 / English / 한국어). Upload an image to get a verdict, a calibrated "AI rate" and a Grad-CAM heatmap; videos go to separate video detectors.

<img src="docs/round_ui.jpg" width="640" alt="Web UI">

*Web UI: an OpenAI-generated face (made by this project) is flagged as AI with AI rate >99%; below are the Grad-CAM maps of both ensemble members.*

## Final model

Mean P(fake) of two models, threshold **0.522**:

| Member | Backbone | Training |
|---|---|---|
| `fm_ft_clip_l14_lora` | CLIP ViT-L/14 (OpenAI) | **LoRA on all layers** (new this round) |
| `fm_ft_dinov2_b` | DINOv2-B (with registers) | last 4 blocks unfrozen (previous round, kept) |

Threshold: 97th percentile on one half of the held-out real photos, so about 97% of real photos are judged real. All numbers below are on the other half and on the other test sets.

## What this round did

1. **LoRA on every layer**: every CLIP ViT-L/14 block gets LoRA on `qkv`, `proj`, `fc1`, `fc2` (r=16, alpha=32, dropout 0.05); the head and final LayerNorm are trained fully; the original weights stay frozen (6.3M trainable parameters). LoRA lr 2e-4, head 1e-3, AdamW, 200 warm-up steps + cosine decay, 5 epochs. After training the LoRA is merged back and saved as a plain timm model, so inference does not need peft.
2. **More weight on real images in the loss**: `BCEWithLogitsLoss(pos_weight=0.4)`, i.e. a real image weighs 2.5x a fake. A false alarm on a real face costs more, which pushes scores down: the previous threshold was 0.715, this one is 0.522, close to 0.5.
3. **More content-matched fakes** (the B-Free idea): a real face goes once through a generator's autoencoder, or one region is inpainted. Content is identical to the original, so the model must learn generator traces instead of shortcuts like who or where. New this round: FLUX.2 and Qwen-Image VAEs, more SD1.5 / SDXL / FLUX VAE round trips, SD2 inpainting up to 1,190 images.
4. **More new-generator data**: 9 more OpenFake shards, about 12,000 more face crops (gpt-image, nano-banana, Imagen 4, Midjourney, Seedream ..., plus matching LAION / Pexels reals).
5. **Camera-original real photos**: unedited photos with faces from the author's own library (camera model in EXIF, no editing software), used as real training data. Kept local, not published.
6. **Test-time augmentation**: horizontal-flip averaging was tested; no gain for this pair, so it is off (`"tta": true` in `ensemble.json` turns it on).

![content-matched fakes](docs/round_aligned.jpg)
*Content-matched fakes: real (top), fake (middle), difference amplified 8x (bottom). VAE round trips leave only faint traces; inpainting changes one region.*

Each epoch samples 15,000 real + 15,000 fake images, balanced across families (equal share per source inside a family); all are 256 px loose face crops, JPEG q95. Augmentation (same for both classes): down-up resize, Gaussian blur, JPEG 50–95, film-look grading, random crop, horizontal flip, light colour jitter.

### Training data

| Real | Images | Fake | Images |
|---|---|---|---|
| FairFace (all ages / skin tones) | 25,940 | SFHQ-T2I (FLUX, SDXL, DALL-E 3) | 10,961 |
| Real video frames | 8,155 | Face-swap video frames (FaceFusion etc.) | 7,430 |
| Real photos (FFHQ / Pexels / PPR10k / Flickr30k / EasyPortrait / DFF reals) | 5,958 | OpenFake (gpt-image, nano-banana, Imagen, MJ, Seedream ...) | 7,064 |
| OpenFake reals (LAION / Pexels) | 5,656 | AI video frames (Veo / Sora / Kling / avatars ...) | 6,804 |
| Reals from StyleGAN datasets | 4,660 | Content-matched VAE round trips (SD1.5 / SDXL / FLUX / FLUX.2 / Qwen-Image) | 6,458 |
| Reddit photography + camera originals | 2,138 | DeepFakeFace (SD text2img / inpainting / InsightFace swap) | 6,000 |
| Reals behind the content-matched fakes | 1,190 | Reddit AI-art subreddits | 5,957 |
| | | StyleGAN2/3 | 3,648 |
| | | Content-matched SD2 inpainting | 1,190 |
| | | OpenAI gpt-image | 897 |

![training loss](docs/round_loss.png)

## Results

![results](docs/round_results.png)

Held-out test sets (people / posts / videos not seen in training; Rapidata never used for training):

| Real: judged real | | AI: caught | |
|---|---|---|---|
| Real photos | 96.7% | OpenAI gpt-image | 100% |
| Real video frames | 95.3% | SFHQ (FLUX / SDXL / DALL-E 3) | 99.9% |
| FairFace | 99.7% | StyleGAN | 99.4% |
| OpenFake reals | 97.8% | DFF SD text2img / inpainting / swap | 97.2% / 94.6% / 76.2% |
| Reddit photography | 97.7% | **Rapidata (newest generators, never trained on)** | **96.4%** |
| StyleGAN dataset reals | 100% | Reddit AI art | 89.7% |
| Reals behind content-matched fakes | 97.0% | OpenFake | 84.1% |
| | | AI video frames | 79.3% |
| | | Face-swap video frames | 63.9% |
| | | Content-matched inpainting | 59.0% |
| | | Content-matched VAE round trip | 41.6% |

![samples](docs/round_samples.jpg)
*Drawn at random from the held-out sets (not hand-picked); the number is the ensemble score, red = above threshold. The first row comes from generators never seen in training; the Midjourney 6 image in the second row is missed.*

- Mean over real sets 97.7% judged real; mean over fake sets 82.1% caught.
- CLIP LoRA alone catches 88.0% on average, but real sets drop to 96.9% (real video frames 93.4%). Averaging with DINOv2 brings false alarms back to the previous level.
- Content-matched VAE fakes are the hardest: same content as the original, only faint autoencoder traces.

**AI rate**: the ensemble score is calibrated with isotonic regression on held-out data (50/50 prior, each test set weighted equally); held-out ECE 3.0% → 0.5%. Score 0.3 ≈ 24% AI rate, 0.5 ≈ 64%, 0.9 ≈ 99%. The verdict still uses the threshold.

![calibration](docs/round_calibration.png)
*Left: score distribution of real and AI images on held-out sets (log scale); right: calibration curve from score to AI rate.*

Full numbers: `results/fm_combo.json` (ensemble selection), `results/arms_race.json` (every model × every test set).

## Usage

Needs [uv](https://docs.astral.sh/uv/). Uses MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU.

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
uv run app.py      # http://127.0.0.1:7860
```

The web UI reads the members from `runs/fm_ens/ensemble.json` and weights from `runs/<member>/model.pt`. This round's weights are not in a Release; train them with the steps below.

## Reproduce this round

Data from earlier rounds (DFF, OpenAI images, modern reals, StyleGAN, video frames) is prepared with `prepare.py`, `gen_openai.py`, `fetch_real.py`, `fetch_gan.py`, `extract_tight.py`; see the commit history. New this round:

```bash
# New generators / reals (Hugging Face)
uv run fetch_hf.py sfhq
uv run fetch_hf.py openfake
uv run fetch_hf.py openfake lists/openfake_shards2.txt openfake2   # the 9 shards added this round
uv run fetch_hf.py fairface
uv run fetch_hf.py rapidata                                        # test only
uv run fetch_hf.py manifest

# Reddit (needs rdt-cli and rdt login)
uv run fetch_reddit.py list && uv run fetch_reddit.py list_real
uv run fetch_reddit.py download && uv run fetch_reddit.py faces

# Content-matched fakes: round 1, then round 2 (new VAEs + more inpainting)
uv run gen_aligned.py
uv run gen_aligned.py --round 2 --train 6000 --test 1000 --max-inpaint 1200 --only vae_flux2 vae_qwen vae_sd15 vae_sdxl vae_flux
uv run gen_aligned.py --round 2 --train 6000 --test 1000 --max-inpaint 1200 --only inpaint

# Optional: camera-original faces from your own library (macOS Photos)
uvx osxphotos query --json --not-edited --only-photos > /tmp/myphotos_query.json
uv run fetch_myphotos.py /tmp/myphotos_query.json

# Training
FM_LORA=16 FM_REAL_WEIGHT=2.5 FM_TAG=_lora uv run fm_detector.py finetune clip_l14 5
uv run fm_detector.py finetune dinov2_b 4   # the live model is from the previous round; retraining on today's data gives slightly different numbers

# Evaluate, pick the ensemble, calibrate
uv run fm_compare.py fm_ft_clip_l14_lora fm_ft_dinov2_b
uv run calibrate.py
uv run arms_race.py
uv run make_figures.py   # figures on this page
```

## Limitations

- Looks at the face region only, not the background or the whole image.
- Face-swap frames (64%) and content-matched VAE fakes (42%) are still weak. Video face swaps rely on the separate video detectors.
- Screenshots and images recompressed many times score lower.
- New generators will need new data and another round of training.

## License

- Code: MIT (see [LICENSE](LICENSE)).
- Third-party datasets are used locally only and **not redistributed**: FairFace (CC BY 4.0), SFHQ-T2I (MIT), OpenFake (CC BY-NC 4.0), Rapidata (CDLA-Permissive-2.0), DFF (OpenRAIL), FFHQ (CC BY-NC-SA 4.0), Reddit posts (Reddit terms), etc.
- Sample images in the `docs/round_*` figures come from Rapidata (CDLA-Permissive-2.0), SFHQ-T2I (MIT), OpenFake (CC BY-NC 4.0), FairFace (CC BY 4.0), Pexels (Pexels License) and OpenAI images generated by this project; shown for illustration only.
- The published detectors used for comparison (UnivFD, NPR, B-Free) are not in this repository; get them from their projects and follow their licenses.

For research and teaching only. Results are not forensic evidence.
