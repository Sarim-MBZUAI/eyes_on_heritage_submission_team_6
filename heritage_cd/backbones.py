"""Frozen image encoders, all exposed the same way: image batch in, spatial feature map out.

spec strings:
  hf:<model id>            any HF ViT with patch tokens (facebook/dinov2-base, facebook/dinov3-vitl16-pretrain-sat493m)
  remoteclip:<file.pt>     RemoteCLIP ViT-L/14 checkpoint for open_clip (chendelong/RemoteCLIP)
  satlas:<file.pth>        SatlasPretrain Aerial Swin-B, single image (allenai/satlas-pretrain, aerial_swinb_si.pth)
"""
import os

import torch

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class Backbone:
    """forward(x): x float in [0, 1], (B, 3, size, size) -> (B, g, g, d) feature map.

    half=True (default) runs the frozen encoder in fp16 for embedding; half=False keeps fp32 weights for
    fine-tuning. blocks() / final_norm() expose the transformer stack so a trainer can unfreeze its tail.
    """

    size = 224

    def __init__(self, mean_std, dev, half=True):
        self.mean = torch.tensor(mean_std[0], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(mean_std[1], device=dev).view(1, 3, 1, 1)
        self.dtype = torch.float16 if half else torch.float32

    def blocks(self):
        raise NotImplementedError

    def final_norm(self):
        raise NotImplementedError

    def norm(self, x):
        return (x - self.mean) / self.std


class HF(Backbone):
    def __init__(self, model_id, dev, half=True):
        from transformers import AutoImageProcessor, AutoModel
        tok = os.environ.get("HUGGINGFACE_TOKEN")
        # satellite DINOv3 ships its own mean/std, which differ from ImageNet's
        pp = AutoImageProcessor.from_pretrained(model_id, token=tok)
        super().__init__((tuple(pp.image_mean), tuple(pp.image_std)), dev, half)
        self.m = AutoModel.from_pretrained(model_id, token=tok).to(dev).eval().to(self.dtype)
        self.patch = self.m.config.patch_size
        self.n_skip = 1 + getattr(self.m.config, "num_register_tokens", 0)
        self.size = (224 // self.patch) * self.patch

    def __call__(self, x):
        g = self.size // self.patch
        h = self.m(pixel_values=self.norm(x).to(self.dtype)).last_hidden_state[:, self.n_skip:]
        assert h.shape[1] == g * g, (h.shape, g)
        return h.reshape(len(x), g, g, -1)

    def blocks(self):
        return self.m.model.layer if hasattr(self.m, "model") else self.m.encoder.layer

    def final_norm(self):
        return self.m.norm if hasattr(self.m, "norm") else self.m.layernorm


class RemoteCLIP(Backbone):
    def __init__(self, ckpt, dev, half=True):
        import open_clip
        super().__init__(CLIP, dev, half)
        self.m = open_clip.create_model("ViT-L-14")
        missing, unexpected = self.m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        assert not unexpected and all(k.startswith("text") or "logit" in k for k in missing), (missing, unexpected)
        self.v = self.m.visual.to(dev).eval().to(self.dtype)
        self.v.output_tokens = True

    def __call__(self, x):
        _, tok = self.v(self.norm(x).to(self.dtype))
        g = self.size // 14
        assert tok.shape[1] == g * g, tok.shape
        return tok.reshape(len(x), g, g, -1)

    def blocks(self):
        return self.v.transformer.resblocks

    def final_norm(self):
        return self.v.ln_post


class Satlas(Backbone):
    """Swin-B trained on 1 m aerial RGB. Input is plain [0, 1]; the stride-16 stage (512 ch) is used."""

    def __init__(self, ckpt, dev, half=True):
        import torchvision
        super().__init__(((0, 0, 0), (1, 1, 1)), dev, half)
        sd = torch.load(ckpt, map_location="cpu")
        prefix = "backbone.backbone."
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        assert sd, "no backbone weights under " + prefix
        swin = torchvision.models.swin_v2_b()
        missing, unexpected = swin.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("head") for k in missing), (missing, unexpected)
        # up to the third stage: stride 16, 512 channels. Kept in float32: Swin-V2's cosine attention
        # overflows in half precision and the SVD downstream then fails on NaNs
        self.f = swin.features[:6].to(dev).eval().float()

    def __call__(self, x):
        h = self.f(self.norm(x).float())  # (B, g, g, 512), channels last already
        assert h.shape[1] == self.size // 16, h.shape
        return h


def load(spec, dev, half=True):
    kind, _, ref = spec.partition(":")
    assert ref, f"bad backbone spec {spec!r}"
    return {"hf": HF, "remoteclip": RemoteCLIP, "satlas": Satlas}[kind](ref, dev, half)
