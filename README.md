**中文** | [English](README.en.md) | [한국어](README.ko.md)

# AI 生成人脸鉴别：DeepFakeFace 复现 + OpenAI 生图 + AI 视频 / 换脸检测

在一台 M2 Max Mac 上，从零训练"真人 vs AI 生成人脸"的检测器，并用实验回答三个问题：

1. **检测器能不能识别没见过的生成器？**（复现教材里的 detector / generator *arms race*）
2. **要多少张新生成器的图，检测器才能学会它？**（以 OpenAI gpt-image 系列为例）
3. **视频能不能测？**（FaceFusion 换脸、Veo / Sora / Kling 等 AI 生成视频）

附带一个 Gradio Web UI（中文 / English / 한국어）：图片或视频放进同一个上传框，自动交给对应的专用模型——图片给出 AI 生成概率 + Grad-CAM 热力图，视频同时跑"AI 生成视频"和"换脸"两个检测器。

![OpenAI 生成样例](docs/openai_samples.jpg)
*上图全部由 gpt-image-1.5 / 2 / 2.5 生成，没有一张是真人。*

## 目录

- [主要结论](#主要结论)
- [视频检测](#视频检测)
- [数据从哪来](#数据从哪来)
- [方法](#方法)
- [快速使用 Web UI](#快速使用-web-ui)
- [完整复现](#完整复现)
- [局限](#局限)
- [许可](#许可)

## 主要结论

| | 结果 |
|---|---|
| 用一种生成方式训练，测没见过的另一种 | 常掉到瞎猜水平（AUC 0.3–0.7） |
| 三种 DFF 生成方式都训练（`all`） | 三类都 > 0.9 AUC，但对 OpenAI 图 AUC 只有 0.44 |
| 加 **50 张** OpenAI 图微调 | OpenAI 图检出率 3% → **83.5%** |
| 加 **500 张** | **98.0%**，再多基本不涨（897 张：98.5%） |
| 现代真实照片误报 | 0–1.4%（Pexels / PPR10k / FFHQ / Flickr30k / EasyPortrait） |
| StyleGAN 人脸（从没见过） | 检出 0% → 补 4,500 张后 **87–99%** |
| FaceFusion 换脸视频 | 宽松构图 + 小模型：识别不了 → **贴脸裁切 + EfficientNet-B4：AUC 0.93–0.98，本地测试的 4 个换脸视频全部识破** |
| AI 生成视频（Veo / Sora / Kling / 数字人） | 留出测试 AUC **0.968**，真视频正确率 93.8%；但**网络转载、重压缩过的视频仍不可靠** |

**一句话：检测器只认识训练时见过的那类生成器；但补几百个新生成器的样本，就能很快追上。视频的关键是"把脸放大看"。**

### 1. 跨生成器测试（DeepFakeFace）

![cross generator](docs/cross_generator.png)

- 只用 SD 文生图训练 → 换脸图 AUC 0.49（瞎猜）。
- 只用换脸训练 → 文生图 AUC **0.305**，比瞎猜还差：模型把扩散模型假图判得比真图还"真"。
- 局部重绘训练的模型迁移最好（文生图 0.965）：它被迫学局部接缝，这种痕迹更通用。
- 测试集为训练中从未出现的人物（按 IMDB-WIKI 人物 ID 切分）。

### 2. OpenAI 学习曲线

![learning curve](docs/learning_curve.png)

固定 200 张从未参与训练的 OpenAI 图 + 3000 张 6 个来源的真实照片做测试：

| 加入 OpenAI 图 | OpenAI AUC | OpenAI 检出率 | 真图正确率 | 旧 DFF 假图检出 |
|---|---|---|---|---|
| 0 | 0.412 | 3.0% | 95.2% | 76.7% |
| 50 | 0.976 | 83.5% | 98.1% | 79.6% |
| 100 | 0.991 | 90.5% | 98.0% | 80.3% |
| 200 | 0.994 | 93.5% | 98.1% | 79.5% |
| 500 | 0.998 | 98.0% | 97.6% | 81.3% |
| 897 | 0.999 | 98.5% | 97.8% | 79.4% |

gpt-image-1.5 / 2 / 2.5 三个版本检出率都约 98%。真图里只有 DFF 的旧 wiki 照片误报偏高（约 10%，因为又小又糊）。完整数据见 [`results/`](results/)。

### 3. StyleGAN（GAN 生成的脸）

图片模型只见过扩散模型（SD、OpenAI），对 StyleGAN 人脸检出率是 0。补 4,500 张 StyleGAN2/3 人脸 + FFHQ 真脸（StyleGAN 就是用 FFHQ 训练的，真脸是最难的反例）后，得到最终图片模型 `all_image`：

| 测试集（从未参与训练） | 补之前 | 补之后 |
|---|---|---|
| thispersondoesnotexist（StyleGAN2） | 0% | 86.8% |
| StyleGAN2-Face | 0.6% | 99.4% |
| StyleGAN3 | 0% | 97.3% |
| 140k Real/Fake（StyleGAN） | 0% | 79.4% |
| OpenAI 图 | 98.5% | 98.5% |
| 真实照片（6 来源） | 97.8% | 97.0%（FFHQ 96.4%） |
| 旧 DFF 假图 | 79.4% | 77.5% |

## 视频检测

视频按帧检测：取若干帧 → YuNet 找最大的人脸 → 打分 → 取所有帧的中位数。

### 失败的路线：宽松构图 + 小模型（4 轮）

最初沿用图片模型的构图（人脸只占画面约 45%，缩到 224 像素后脸只剩约 100 像素宽），用 EfficientNet-B0 在视频帧上微调，做了 4 轮：

| 轮次 | 加了什么 | 结果 |
|---|---|---|
| 1 | 公开 AI 视频 / 换脸数据集的帧，真视频只有 FF++ | 什么视频都判"假"：真视频只有 59% 判对 |
| 2 | 加 CelebV-HQ 真视频 | 真视频 88% 判对，但本地 AI 视频又全部漏掉 |
| 3 | 用本机 FaceFusion 生成约 300 个换脸视频（换上 OpenAI 生成的脸） | 留出测试 AUC 0.80–0.92，但本地钢铁侠换脸仍是 0 分 |
| 4 | 换上真人照片（FFHQ）、换进电影片段，加"电影感"增强 | 新换脸只有 AUC 0.67–0.77；钢铁侠仍漏判 |

结论：高质量换脸的破绽（边缘、皮肤纹理、眼睛和牙齿）非常细，100 像素的脸里基本看不到。第 3 轮的好成绩还部分来自"换上去的是 AI 脸"这条捷径。

### 有效的路线：贴脸裁切 380 像素 + EfficientNet-B4

改成换脸检测研究的标准做法：只裁人脸框的 1.3 倍、放大到 380×380（`extract_tight.py`），用更大的 EfficientNet-B4（`train_swap.py`），每个视频取 16 帧。真假视频成对生成、同一画质编码，阈值在验证集上选。

**换脸检测器** `runs_swap/b4`（1,014 个视频：421 真 / 593 换脸）：

| 测试（未参与训练的人物 / 视频） | 结果 |
|---|---|
| FaceFusion hyperswap 1a/1b/1c、alphaface、inswapper | AUC 0.93–0.98，检出 75–100% |
| DF40 inswap / simswap | AUC 0.97–0.98，检出 100% |
| FF++（Deepfakes、FaceSwap 等 5 种） | AUC 0.80–0.96 |
| 真视频 | 90% 判对 |
| **本地钢铁侠换脸视频 ×4**（用作者本人照片换，完全没参与训练） | **4/4 识破（中位数 1.00）**；原片 0.00 |
| 原片重新编码（q95/80/60、720p） | 全部判真——模型认的是换脸，不是"二次压缩" |
| 换脸视频再压到 720p / crf 28 | 仍判假（0.98） |

**AI 生成视频检测器** `runs_aivid/b4`（Veo 2/3/3.1、Sora 1/2、Kling、Wan、Pika、Ray2、Hunyuan、6 种数字人 / 说话头、HeyGen；共 2,519 个假视频 + 2,584 个真视频，含"网络重压缩"副本）：

| 版本 | 改动 | 留出测试 AUC | 真视频判对 | 假视频检出 |
|---|---|---|---|---|
| v1 | 初版 | 0.80 | 87% | 57% |
| v2 | 补 TalkingHeadBench + CelebV 真视频 | 0.96 | 84% | 92% |
| v3 | 所有视频加一份 YouTube 式重压缩副本（360–720p，H.264 / VP9） | 0.96 | 84% | 94% |
| **v4（采用）** | 真视频翻倍（+600 个 CelebV） | **0.968** | **93.8%** | 88% |

（四个版本在同一个测试集上比较：991 个留出视频，含重压缩副本。四个模型集成没有超过 v4。）

**未解决**：两个网上转载的 AI 视频（Wikimedia 转码的 Veo 3 片段、YouTube 上的 AI 假新闻）在四个版本里的分数都在阈值附近摇摆——重压缩后剩下的痕迹太少。另外，混合了真人画面的剪辑视频，中位数分数会被真画面拉低。

## 数据从哪来

共用到 3 类数据，总计约 15 万张图。所有图片统一处理成 256×256 JPEG（质量 95）。

### A. DeepFakeFace（DFF）——主训练集

- 来源：[OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace)（Hugging Face，OpenRAIL 许可，4 个 zip 共约 5GB）
- 内容：

| zip | 类别 | 数量 | 生成方式 |
|---|---|---|---|
| `wiki.zip` | 真 | 30,000 | IMDB-WIKI 名人照片 |
| `text2img.zip` | 假 | 30,000 | Stable Diffusion v1.5 文生图 |
| `inpainting.zip` | 假 | 30,000 | SD Inpainting（局部重绘真图） |
| `insight.zip` | 假 | 30,000 | InsightFace 换脸 |

- 四个 zip 结构相同：`<类别>/<00–99>/<人物ID>_<生日>_<拍摄年>.jpg`，假图与真图同名一一对应。
- 子文件夹编号 = 人物 ID 的末两位，同一个人永远在同一个文件夹，所以按文件夹切分不会泄漏：**00–79 训练、80–89 验证、90–99 测试**。
- 原始真图尺寸不一；DFF 作者把它们拉伸成 512×512 后生成假图，所以本项目统一拉伸到 256×256，真假几何一致。
- 完整清单：[`data_lists/dff_manifest.csv`](data_lists/dff_manifest.csv)。

### B. OpenAI 生成人脸——本项目自己生成

- 用 OpenAI Images API（`/v1/images/generations`，1024×1024，默认 medium 质量）生成。
- 3 个模型：gpt-image-2（434 张）、gpt-image-1.5（332 张）、gpt-image-2.5（331 张），共 **1,097 张**（请求 1,100 张，3 张被 API 以 400 拒绝）。
- 提示词由程序随机组合（固定随机种子 0，可完全复现）：

```
Realistic photograph of a {年龄} {人种} {性别} {场景}, face clearly visible, upper body, {拍摄风格}. No text, no watermark.
```

  - 年龄 6 种（young … elderly）× 人种 6 种 × 性别 2 种 × 场景 8 种（红毯、发布会、街拍、办公室、体育赛事、演播室……）× 风格 5 种（2008 年数码相机、通讯社照片、手机随拍、专业头像、老扫描照片）。
  - 场景和风格刻意模仿 IMDB-WIKI 那种名人新闻照，让真假图题材接近。
- 原图为 PNG（约 1024–1300px），内含 OpenAI 写入的 C2PA 内容凭证；训练时居中裁成正方形再缩到 256×256。
- 下载：[Releases v0.1](../../releases/tag/v0.1)

| 文件 | 内容 |
|---|---|
| `openai_faces_256.zip`（26MB） | 1,097 张 256×256 JPEG（训练用的版本） |
| `openai_faces_orig_part1.zip` / `part2.zip`（各 1.1GB） | 原始 PNG |
| `openai_meta.json` | 每张图的模型与提示词 |

### C. 现代真实人像——防止"清晰 = 假"的捷径

DFF 的真图大多又旧又糊，而 OpenAI 图清晰现代。只用 DFF 真图，模型可能学成"清晰的现代照片就是假的"。所以额外加了 5 个真实照片集：

| 来源 | 内容 | 用量 | 取法 |
|---|---|---|---|
| [FFHQ](https://huggingface.co/datasets/bitmind/ffhq-256) | Flickr 高清人脸（对齐裁切） | 4,375 | `bitmind/ffhq-256` 第 1 个 parquet 分片 |
| [Pexels Portrait](https://huggingface.co/datasets/gaunernst/pexels-portrait) | 专业人像摄影 | 3,000 | 只取 Pexels ID < 14,000,000（约 2023 年前上传，避开 AI 图），随机抽 |
| [PPR10k](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k) | 人像写真原片（未修图） | 3,000 | 每个样本的 `before.jpg`，随机抽 |
| [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) | 日常生活照 | 1,458 | zip 中随机抽 3,000 个文件，其中约一半是 macOS `._` 元数据文件被跳过 |
| [EasyPortrait](https://huggingface.co/datasets/gofixyourself/EasyPortrait) | 摄像头自拍（视频会议场景） | 3,000 | `data/test.zip` 中随机抽 |

- 随机抽样都用固定种子 0。非正方形图片居中裁切；竖图从顶部裁，保住人脸。
- 这些照片含可识别的真人，且部分许可不允许再分发，**本仓库不上传**；用到的每一张的 ID 列在 [`data_lists/`](data_lists/)。

### D. StyleGAN 人脸（`fetch_gan.py`）

| 来源 | 生成器 | 用量 |
|---|---|---|
| [javi22/this-person-does-not-exist-10k](https://huggingface.co/datasets/javi22/this-person-does-not-exist-10k) | StyleGAN2 | 1,500 |
| [gojay/StyleGAN2-Face](https://huggingface.co/datasets/gojay/StyleGAN2-Face) | StyleGAN2 | 1,000 |
| [34data/StyleGAN3_1024x1024_images](https://huggingface.co/datasets/34data/StyleGAN3_1024x1024_images) | StyleGAN3 | 1,000 |
| [TheKernel01/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) | StyleGAN 假脸 + FFHQ 真脸 | 各 1,000 |

每个来源随机 20% 做测试。

### E. 视频

| 类别 | 来源 | 获取方式 |
|---|---|---|
| 换脸（自己生成） | 本机 FaceFusion 3.9：hyperswap 1a/1b/1c、alphaface、inswapper；把 FFHQ 真人脸换进 CelebV-HQ 访谈片段和原始 YouTube 视频（含电影、剧集）的 4 秒片段 | `gen_faceswap.py --targets clips/raw --faces ffhq --keep`；真片和换脸片用同一画质各编码一次 |
| 换脸（公开） | [FaceForensics++ C23](https://huggingface.co/datasets/bitmind/FaceForensicsC23)、[DF40](https://huggingface.co/datasets/ohjoonhee/DF40-Fake-Videos) | `fetch_public_videos.py`，只按需下载 zip 里的部分视频（HTTP range） |
| AI 生成视频 | [34data gen-videos](https://huggingface.co/datasets/34data/gen-videos-veo3)（Veo 3 / Sora 2 / Kling / Wan 2）、[GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench)（Sora / 可灵）、[Rapidata](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo3) 偏好数据集（Veo 2/3/3.1、Sora、Pika、Ray2、Hunyuan、Wan、Alpha） | `fetch_rapidata.py`、`fetch_aivid_videos.py`；只保留有可检测人脸的视频 |
| 数字人 / 说话头 | [TalkingHeadBench](https://huggingface.co/datasets/luchaoqi/TalkingHeadBench)（LivePortrait、Hallo 1/2、AniPortrait ×2、EmoPortrait）、DF40 HeyGen | 同上 |
| 真视频 | FF++ 原片、[CelebV-HQ](https://huggingface.co/datasets/mengtingwei/celebv-HQ-raw-videos) 原始视频片段与[访谈片段](https://huggingface.co/datasets/saeed-5959/celebv_hq_head_talking) | 同上 |
| 重压缩副本 | 以上每个视频一份：360/480/720p，H.264 crf 26–34 或 VP9 | `recompress_videos.py` |

切分按身份 / 提示词分组（一个真片和它的换脸版本在同一组），测试集里的人物和视频都没参与训练。视频文件和抽出的人脸图都不分发。

## 方法

| 步骤 | 文件 | 说明 |
|---|---|---|
| 数据 | `prepare.py` | 下载 DFF → 256×256 → 按人物文件夹切分，写 `data/manifest.csv` |
| 训练 | `train.py` | EfficientNet-B0（timm，ImageNet 预训练）二分类，AdamW + OneCycle，lr 3e-4，batch 64，4 epoch，取验证 AUC 最好的一轮。真假两类做相同的随机降采样 / 模糊 / JPEG 压缩 / 裁切 / 翻转 / 调色，防止靠清晰度作弊。每个 epoch 用全部真图 + 等量随机抽的假图，保持类别平衡 |
| 实验 | `run_all.sh` | 4 个模型：`t2i`、`inpaint`、`insight`、`all` |
| OpenAI 假图 | `gen_openai.py` | 见上文 B |
| 现代真图 | `fetch_real.py` | 见上文 C |
| 零样本测试 | `eval_openai.py` | DFF 模型直接测 OpenAI 图 |
| 学习曲线 | `openai_curve.py` | 1,097 张 OpenAI 图随机留 200 张做测试，其余 897 张作训练池。从 `all` 出发，每次用 N 张（0/50/100/200/500/897）微调 3 epoch，lr 1e-4。每 epoch：约 6,000 张真图（6 个来源各最多 1,000）+ 3,000 张 DFF 假图 + 3,000 张 OpenAI 图（N 张重复采样）。测试真图 = 6 个来源各 500 张（与训练不重叠） |
| StyleGAN | `gan_finetune.py` | 从 `all_openai` 出发微调 4 epoch：真图 = 6 来源 + 全部非测试 FFHQ + 140k 真脸；假图 = 4 个 StyleGAN 来源均分 + DFF + OpenAI → `runs/all_image` |
| 视频（宽松构图，旧） | `video_scan.py`、`fetch_video.py`、`video_finetune.py` | 上文"失败的路线" |
| 视频贴脸裁切 | `extract_tight.py` | 每个视频 16 帧，人脸框 1.3 倍，380×380 |
| B4 检测器 | `train_swap.py` | EfficientNet-B4，batch 16，lr 1e-4，5 epoch × 1,500 步；按类别、按来源均衡采样；10% 训练组做验证，选最好的 epoch 和阈值；`--task swap` 或 `--task aivid`；训练时加"电影感"增强（暗化、调色、颗粒） |
| 换脸视频生成 | `gen_faceswap.py` | 调用本机 FaceFusion 3.9 headless，5 个模型轮换，pixel boost 256/512，mask = box + occlusion（xseg_3） |
| 视频扫描 | `scan_swap.py` | `--model runs_swap/b4` 或 `runs_aivid/b4`，32 帧，输出接触表 |
| Web UI | `app.py` | 一个上传框：图片 → `all_image` + Grad-CAM；视频 → 两个 B4 检测器，任一超过阈值即判可疑并说明更像哪一类；三种界面语言 |

## 快速使用 Web UI

需要 [uv](https://docs.astral.sh/uv/)。Apple Silicon 自动用 MPS，NVIDIA 用 CUDA，否则用 CPU。

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
# 下载模型权重，在项目根目录解压（得到 runs*/…/model.pt）
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.2/models.zip
unzip models.zip
uv run app.py      # 打开 http://127.0.0.1:7860，在"一键检测"里上传图片或视频
```

`models.zip`（v0.2）内含：

| 目录 | 用途 |
|---|---|
| `runs/all_image/` | **图片**：DFF 三类 + OpenAI + StyleGAN，Web UI 默认 |
| `runs_swap/b4/` | **视频换脸**检测器（EfficientNet-B4，含阈值） |
| `runs_aivid/b4/` | **AI 生成视频**检测器（EfficientNet-B4 v4，含阈值） |
| `runs/all_openai/`、`runs/all/`、`runs/t2i/` `runs/inpaint/` `runs/insight/`、`runs/all_video/` | 对照 / arms race 实验 / 旧的宽松构图视频模型 |

## 完整复现

按顺序运行。时间为 M2 Max（64GB）上的实测，磁盘约需 20GB。

```bash
# 0. 环境
uv sync

# 1. DFF：下载 4 个 zip（5GB）→ 缩放 → 切分（约 10 分钟 + 下载时间）
uv run prepare.py

# 2. 4 个 DFF 模型（约 3 小时）
./run_all.sh
#    结果：runs/<name>/results.json，日志：logs/

# 3. OpenAI 生成图
#    方式 a：直接用发布的图（推荐，结果完全一致）
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.1/openai_faces_256.zip
mkdir -p data/img256 data/openai
unzip openai_faces_256.zip -d data/img256 && mv data/img256/meta.json data/openai/
#    方式 b：自己生成（约 2 小时，需 API key；OpenAI 输出有随机性，图不会完全一样）
export OPENAI_API_KEY=sk-...
#    export OPENAI_BASE_URL=...   # 用兼容 OpenAI 的中转站时设置
uv run gen_openai.py --n 100 --models gpt-image-2            # 第 0–99 张
uv run gen_openai.py --n 1100 --start 100 \
    --models gpt-image-1.5 gpt-image-2 gpt-image-2.5          # 第 100–1099 张，三模型轮换

# 4. 现代真图（约 30 分钟，下载约 10GB，处理完自动删除原始包）
uv run fetch_real.py ffhq pexels ppr10k flickr30k easyportrait

# 5. DFF 模型零样本测 OpenAI 图（几分钟）
uv run eval_openai.py

# 6. 学习曲线（约 40 分钟）→ runs_openai/curve.json、runs_openai/n*.pt
uv run openai_curve.py

# 7. 把最终模型放进 Web UI
mkdir -p runs/all_openai && cp runs_openai/n897.pt runs/all_openai/model.pt
uv run app.py

# 8. StyleGAN（约 1 小时）→ runs/all_image
uv run fetch_gan.py
uv run gan_finetune.py

# 9. 视频数据（下载约 15GB；换脸生成在 M2 Max 上约 5 小时，需本机 FaceFusion 3.x）
uv run gen_faceswap.py --clips 120 --targets raw   --faces ffhq --tag hq_raw  --keep
uv run gen_faceswap.py --clips 120 --targets clips --faces ffhq --tag hq_clip --keep
uv run fetch_public_videos.py
uv run fetch_rapidata.py
uv run fetch_aivid_videos.py            # zips thb celebv celebvclips celebv2
uv run recompress_videos.py aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv

# 10. 换脸检测器（约 2 小时）
uv run extract_tight.py --lists hq_raw.csv hq_clip.csv public.csv --out swap_frames.csv
uv run train_swap.py

# 11. AI 生成视频检测器（约 2.5 小时）
uv run extract_tight.py --lists aivid.csv rapidata.csv hq_clip.csv hq_raw.csv public.csv recomp.csv --out aivid_frames.csv
uv run train_swap.py --frames aivid_frames.csv --task aivid --out runs_aivid/b4

# 12. 扫描视频
uv run scan_swap.py your_video.mp4                       # 换脸
uv run scan_swap.py --model runs_aivid/b4 your_video.mp4 # AI 生成
```

注意：

- 随机种子都固定为 0，但 GPU 训练有非确定性，复现数字会有小幅波动（通常 ±1%）。
- Pexels 原图来自 Pexels CDN，个别照片可能已被作者删除，会自动跳过。
- `gen_openai.py` 会跳过已存在的图片，中断后重跑即可续传。

## 局限

- OpenAI 测试图与训练图出自同一套提示词模板；自拍、合影、经过微信压缩的图准确率可能更低。
- 只覆盖人脸。图片：SD1.5、InsightFace、gpt-image、StyleGAN2/3；Midjourney、Flux、Gemini 等未测试。
- 视频：逐帧模型，不利用时序信息（闪烁、口型与声音是否一致）。网络转载、多次重压缩的 AI 视频仍不可靠；真视频仍有约 6% 被误判。
- 换脸检测器测试用的 FaceFusion 视频和训练数据出自同一个工具（虽然人物、画面、换上的脸都不同）；DeepFaceLab 等其他工具只在 FF++ / DF40 上测过。
- 新的生成器出现后，需要补样本重新微调——这正是 arms race。

## 许可

- 代码：MIT（见 [LICENSE](LICENSE)）。
- OpenAI 生成图（Release 中的 `openai_faces_*` 与 `docs/openai_samples.jpg`）：本项目用 OpenAI API 生成，按 CC BY 4.0 共享，请注明出处。图中人物均不存在。
- 第三方数据集：本仓库**不分发**，只给文件清单。各自许可：DFF（OpenRAIL）、FFHQ（CC BY-NC-SA 4.0）、EasyPortrait（CC BY-SA 4.0）、Pexels（Pexels License）、Flickr30k / PPR10k（仅研究用途）。视频数据集（FF++、DF40、CelebV-HQ、TalkingHeadBench、GenVidBench、Rapidata、34data、StyleGAN 各集）同样只在本地使用，请遵循各自条款；FaceForensics++ 官方版本需同意其使用条款。

本项目仅供研究与教学，检测结果不能作为鉴定依据。
