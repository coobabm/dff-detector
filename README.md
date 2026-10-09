**中文** | [English](README.en.md) | [한국어](README.ko.md)

# AI 生成人脸鉴别：DeepFakeFace 复现 + OpenAI 生图检测

在一台 M2 Max Mac 上，从零训练一个"真人照片 vs AI 生成人脸"的检测器，并用实验回答两个问题：

1. **检测器能不能识别没见过的生成器？**（复现教材里的 detector / generator *arms race*）
2. **要多少张新生成器的图，检测器才能学会它？**（以 OpenAI gpt-image 系列为例）

附带一个 Gradio Web UI：上传人脸图 → AI 生成概率 + Grad-CAM 热力图。

![OpenAI 生成样例](docs/openai_samples.jpg)
*上图全部由 gpt-image-1.5 / 2 / 2.5 生成，没有一张是真人。*

## 目录

- [主要结论](#主要结论)
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
| AI 视频（Veo 3、ModelScope、FaceFusion 换脸） | **基本识别不了**——训练里没有这类数据 |

**一句话：检测器只认识训练时见过的那类生成器；但补几百张新生成器的样本，就能很快追上。**

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

### 3. 视频

每秒取 2 帧 → YuNet 人脸检测 → 打分。最好的模型对 Veo 3、ModelScope、AI 假新闻视频、FaceFusion 3.9 换脸视频的中位数分数都 < 0.3，基本漏判。原因：没有视频生成器 / 新一代换脸工具的训练数据，且 H.264 压缩抹掉了细节痕迹。

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
| 视频 | `video_scan.py` | 每秒 2 帧，YuNet 找人脸，按 DFF 构图（人脸约占宽度 45%）裁切后打分，输出接触表 |
| Web UI | `app.py` | Gradio：上传 → 概率 + Grad-CAM；可切换模型；跨生成器结果表 |

## 快速使用 Web UI

需要 [uv](https://docs.astral.sh/uv/)。Apple Silicon 自动用 MPS，NVIDIA 用 CUDA，否则用 CPU。

```bash
git clone https://github.com/coobabm/dff-detector.git
cd dff-detector
uv sync
# 下载模型权重，在项目根目录解压（得到 runs/*/model.pt）
curl -LO https://github.com/coobabm/dff-detector/releases/download/v0.1/models.zip
unzip models.zip
uv run app.py      # 打开 http://127.0.0.1:7860，模型选 all_openai
```

`models.zip` 内含 5 个模型：

| 目录 | 训练数据 | 用途 |
|---|---|---|
| `runs/all_openai/` | DFF 三类 + 897 张 OpenAI + 6 来源真图 | **推荐**，综合最好 |
| `runs/all/` | DFF 三类 | 对照 |
| `runs/t2i/` `runs/inpaint/` `runs/insight/` | DFF 单类 | arms race 实验 |

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

# 8. （可选）扫描视频
uv run video_scan.py your_video.mp4   # → scans/<视频名>.jpg / .json
```

注意：

- 随机种子都固定为 0，但 GPU 训练有非确定性，复现数字会有小幅波动（通常 ±1%）。
- Pexels 原图来自 Pexels CDN，个别照片可能已被作者删除，会自动跳过。
- `gen_openai.py` 会跳过已存在的图片，中断后重跑即可续传。

## 局限

- OpenAI 测试图与训练图出自同一套提示词模板；自拍、合影、经过微信压缩的图准确率可能更低。
- 只覆盖人脸；只覆盖 SD1.5、InsightFace、gpt-image 系列。Midjourney、Flux、Gemini 等未测试。
- 单帧模型，不利用视频时序信息。
- 新的生成器出现后，需要补样本重新微调——这正是 arms race。

## 许可

- 代码：MIT（见 [LICENSE](LICENSE)）。
- OpenAI 生成图（Release 中的 `openai_faces_*` 与 `docs/openai_samples.jpg`）：本项目用 OpenAI API 生成，按 CC BY 4.0 共享，请注明出处。图中人物均不存在。
- 第三方数据集：本仓库**不分发**，只给文件清单。各自许可：DFF（OpenRAIL）、FFHQ（CC BY-NC-SA 4.0）、EasyPortrait（CC BY-SA 4.0）、Pexels（Pexels License）、Flickr30k / PPR10k（仅研究用途）。

本项目仅供研究与教学，检测结果不能作为鉴定依据。
