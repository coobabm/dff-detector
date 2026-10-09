# AI 生成人脸鉴别：DeepFakeFace 复现 + OpenAI 生图检测

在一台 M2 Max Mac 上，从零训练一个"真人照片 vs AI 生成人脸"的检测器，并用实验回答两个问题：

1. **检测器能不能识别没见过的生成器？**（复现教材里的 detector / generator *arms race*）
2. **要多少张新生成器的图，检测器才能学会它？**（以 OpenAI gpt-image 系列为例）

附带一个 Gradio Web UI：上传人脸图 → AI 生成概率 + Grad-CAM 热力图。

![OpenAI 生成样例](docs/openai_samples.jpg)
*上图全部由 gpt-image-1.5 / 2 / 2.5 生成，没有一张是真人。*

## 主要结论

| | 结果 |
|---|---|
| 用一种生成方式训练，测没见过的另一种 | 常掉到瞎猜水平（AUC 0.3–0.7） |
| 三种 DFF 生成方式都训练（`all`） | 三类都 > 0.9 AUC，但对 OpenAI 图 AUC 只有 0.44 |
| 加 **50 张** OpenAI 图微调 | OpenAI 图检出率 3% → **83.5%** |
| 加 **500 张** | **98.0%**，再多基本不涨（897 张：98.5%） |
| 现代真实照片误报 | 1–2%（Pexels / PPR10k / FFHQ / Flickr30k / EasyPortrait） |
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

gpt-image-1.5 / 2 / 2.5 三个版本检出率都约 98%。完整数据见 [`results/`](results/)。

### 3. 视频

每秒取 2 帧 → YuNet 人脸检测 → 打分。最好的模型对 Veo 3、ModelScope、AI 假新闻视频、FaceFusion 3.9 换脸视频的中位数分数都 < 0.3，基本漏判。原因：没有视频生成器 / 新一代换脸工具的训练数据，且 H.264 压缩抹掉了细节痕迹。

## 方法

| 步骤 | 文件 | 说明 |
|---|---|---|
| 数据 | `prepare.py` | [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace)：3 万真人 + SD1.5 文生图 / SD Inpainting / InsightFace 换脸各 3 万。统一 256×256；按人物 ID 文件夹切 train 00–79 / val 80–89 / test 90–99，防止"认人"泄漏 |
| 训练 | `train.py` | EfficientNet-B0（ImageNet 预训练）二分类，4 epoch，MPS。真假两类做相同的模糊 / 降采样 / JPEG 增强，防止靠清晰度作弊 |
| 实验 | `run_all.sh` | 4 个模型：`t2i`、`inpaint`、`insight`、`all` |
| OpenAI 假图 | `gen_openai.py` | gpt-image-1.5 / 2 / 2.5 轮流生成 1,100 张（成功 1,097），提示词随机组合年龄 × 性别 × 人种 × 场景 × 拍摄风格 |
| 现代真图 | `fetch_real.py` | FFHQ、Pexels（仅 2023 年前上传）、PPR10k、Flickr30k、EasyPortrait，各取约 3000 张，防止模型学成"清晰现代照片 = 假" |
| 零样本测试 | `eval_openai.py` | DFF 模型直接测 OpenAI 图 |
| 学习曲线 | `openai_curve.py` | 从 `all` 出发，加 N 张 OpenAI 图 + DFF 假图 + 6 来源真图微调 3 epoch |
| 视频 | `video_scan.py` | YuNet 人脸检测 + 逐帧打分，输出接触表 |
| Web UI | `app.py` | Gradio：上传 → 概率 + Grad-CAM；可切换模型；跨生成器结果表 |

## 运行

需要 [uv](https://docs.astral.sh/uv/)。Apple Silicon 用 MPS，NVIDIA 用 CUDA，否则 CPU。

```bash
uv sync

# 只想用 Web UI：从 Releases 下载 models.zip，在项目根目录解压（得到 runs/*/model.pt）
uv run app.py                 # http://127.0.0.1:7860

# 完整复现
uv run prepare.py             # 下载解压 DFF 后（data/raw/*.zip）
./run_all.sh                  # 4 个 DFF 模型，M2 Max 约 3 小时
OPENAI_API_KEY=... uv run gen_openai.py --n 1100 --start 100 \
    --models gpt-image-1.5 gpt-image-2 gpt-image-2.5
uv run fetch_real.py pexels ppr10k flickr30k easyportrait
uv run eval_openai.py
uv run openai_curve.py
uv run video_scan.py some_video.mp4
```

DFF 压缩包下载：`https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/{wiki,text2img,inpainting,insight}.zip` → `data/raw/`。FFHQ 来自 [bitmind/ffhq-256](https://huggingface.co/datasets/bitmind/ffhq-256) 第一个 shard。

## 模型权重

[Releases](../../releases) 里的 `models.zip`（约 80MB），内含：

| 目录 | 训练数据 | 用途 |
|---|---|---|
| `runs/all_openai/` | DFF 三类 + 897 张 OpenAI + 6 来源真图 | **推荐**，综合最好 |
| `runs/all/` | DFF 三类 | 对照 |
| `runs/t2i/` `runs/inpaint/` `runs/insight/` | DFF 单类 | arms race 实验 |

## 局限

- OpenAI 测试图与训练图出自同一套提示词模板；自拍、合影、经过微信压缩的图准确率可能更低。
- 只覆盖人脸；只覆盖 SD1.5、InsightFace、gpt-image 系列。Midjourney、Flux、Gemini 等未测试。
- 单帧模型，不利用视频时序信息。
- 新的生成器出现后，需要补样本重新微调——这正是 arms race。

## 许可

代码 MIT。数据集各自遵循原许可（DFF：OpenRAIL；FFHQ：CC BY-NC-SA 4.0；EasyPortrait：CC BY-SA 4.0；Pexels：Pexels License；Flickr30k / PPR10k：仅研究用途），本仓库**不分发**任何数据集图片。`docs/openai_samples.jpg` 为本项目用 OpenAI API 生成。

本项目仅供研究与教学，检测结果不能作为鉴定依据。
