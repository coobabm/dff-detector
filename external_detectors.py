"""Published, pretrained AI-image detectors, run zero-shot on the uploaded image (not our face crop).

  UnivFD (Ojha et al., CVPR 2023)  frozen CLIP ViT-L/14 + linear layer, trained on ProGAN   MIT
  NPR    (Tan et al., CVPR 2024)   ResNet on neighbouring-pixel residuals (up-sampling traces) no licence stated
  B-Free (Guillaro et al., CVPR 2025) DINOv2-B (registers), 5 crops of 504 px, bias-free training  non-commercial

Their code and weights are not part of this repository (licences). Put them under external/ (gitignored),
or point DFF_EXTERNAL at another folder:
  external/univfd/fc_weights.pth                       from WisconsinAIVision/UniversalFakeDetect/pretrained_weights
  external/univfd/clip_vit_l14.safetensors             optional, CLIP weights for offline servers (save_clip())
  external/npr/resnet.py, external/npr/NPR.pth         from chuangchuangtan/NPR-DeepfakeDetection (networks/resnet.py)
  external/bfree/wrapper5crops.py                      from grip-unina/B-Free code/networks/
  external/bfree/BFREE_dino2reg4/model_epoch_best.pth  from grip.unina.it B-Free weights zip
UnivFD's CLIP backbone comes from timm (vit_large_patch14_clip_quickgelu_224.openai, same numbers as
OpenAI's TorchScript file), so no TorchScript is executed. All checkpoints load with weights_only=True.
Each detector returns P(fake) = sigmoid(logit); the authors' threshold is 0.5.
"""
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.transforms as T

EXT = Path(os.environ.get("DFF_EXTERNAL", Path(__file__).parent / "external"))
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FILES = {
    "univfd": ["univfd/fc_weights.pth"],
    "npr": ["npr/resnet.py", "npr/NPR.pth"],
    "bfree": ["bfree/wrapper5crops.py", "bfree/BFREE_dino2reg4/model_epoch_best.pth"],
}
NAMES = {"univfd": "UnivFD (CVPR'23)", "npr": "NPR (CVPR'24)", "bfree": "B-Free (CVPR'25)"}
# Decision thresholds on P(fake). Authors' default is 0.5; B-Free at 0.5 flags 8% of our held-out real
# photos and 23% of the StyleGAN-set reals, so it uses 0.924 (logit 2.5 = 97% of held-out real faces pass).
THRESHOLDS = {"univfd": 0.5, "npr": 0.5, "bfree": 0.924}
MAX_SIDE = 1024  # CPU budget: larger uploads are centre-cropped (not resized: resizing erases the traces NPR reads)


# Switched off in one-click detection (files may stay installed). B-Free: ~9 s per image on a CPU
# server and weak on faces (28-56% caught). UnivFD: scores ~0 on nearly every upload (faces AUC 0.66).
# Override with env DFF_EXTERNAL_DISABLE="" (or a comma list).
DISABLED = {k for k in os.environ.get("DFF_EXTERNAL_DISABLE", "bfree,univfd").split(",") if k}


def available():
    return [k for k, files in FILES.items() if k not in DISABLED and all((EXT / f).exists() for f in files)]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _limit(im):
    w, h = im.size
    if max(w, h) <= MAX_SIDE:
        return im
    cw, ch = min(w, MAX_SIDE), min(h, MAX_SIDE)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))


def _even(im):
    w, h = im.size  # NPR's 2x2 residual needs even sides (as in the authors' test code)
    return im.crop((0, 0, w - w % 2, h - h % 2))


class UnivFD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        # -> 768-d CLIP image embedding. Offline servers: weights from external/univfd/clip_vit_l14.safetensors.
        local = EXT / "univfd/clip_vit_l14.safetensors"
        self.clip = timm.create_model("vit_large_patch14_clip_quickgelu_224.openai", pretrained=not local.exists())
        if local.exists():
            from safetensors.torch import load_file

            self.clip.load_state_dict(load_file(local))
        self.fc = torch.nn.Linear(768, 1)
        self.fc.load_state_dict(torch.load(EXT / "univfd/fc_weights.pth", map_location="cpu", weights_only=True))

    def forward(self, x):
        return self.fc(self.clip(x))


@lru_cache(maxsize=None)
def load(name, device="cpu"):
    """(model, preprocess) for one detector."""
    if name == "univfd":
        model = UnivFD()
        tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    elif name == "npr":
        model = _module("npr_resnet", EXT / "npr/resnet.py").resnet50(num_classes=1)
        sd = torch.load(EXT / "npr/NPR.pth", map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()})
        tf = T.Compose([T.Lambda(_limit), T.Lambda(_even), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    elif name == "bfree":
        import timm

        wrapper = _module("bfree_wrapper5crops", EXT / "bfree/wrapper5crops.py").Wrapper5crops
        model = wrapper(timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", num_classes=1, pretrained=False), 504)
        sd = torch.load(EXT / "bfree/BFREE_dino2reg4/model_epoch_best.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(sd["model"])
        tf = T.Compose([T.Lambda(_limit), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    else:
        raise ValueError(name)
    return model.to(device).eval(), tf


def save_clip():
    """Write timm's CLIP ViT-L/14 weights to external/univfd/ for servers without Hugging Face access."""
    import timm
    from safetensors.torch import save_file

    m = timm.create_model("vit_large_patch14_clip_quickgelu_224.openai", pretrained=True)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, EXT / "univfd/clip_vit_l14.safetensors")


@torch.no_grad()
def score(name, image, device="cpu"):
    """P(fake) for one PIL RGB image."""
    model, tf = load(name, device)
    logit = model(tf(image).unsqueeze(0).to(device)).flatten()[0]
    return torch.sigmoid(logit).item()
