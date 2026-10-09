#!/bin/bash
# Train-on-A / test-on-all experiments, then the combined model.
set -e
cd "$(dirname "$0")"
uv run python train.py --fakes text2img --name t2i > logs/t2i.txt 2>&1
uv run python train.py --fakes inpainting --name inpaint > logs/inpaint.txt 2>&1
uv run python train.py --fakes insight --name insight > logs/insight.txt 2>&1
uv run python train.py --fakes text2img inpainting insight --name all > logs/all.txt 2>&1
