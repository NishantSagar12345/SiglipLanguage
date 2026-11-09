"""SigLIP image-text scoring with visual-token reduction.

Adapted from efficient-vit-bench/src/zeroshot.py (validated on a Kaggle T4).
The scoring rule is the one used in ``SiglipModel.forward`` (transformers 4.46 and 5.x):

    image_embeds = vision pooler_output (MAP head)       text_embeds = text pooler_output
    both L2-normalised
    logits_per_image = image_embeds @ text_embeds.T * exp(logit_scale) + logit_bias

Image embeddings come from ``TokenReducedSiglipVision`` (unchanged from the
efficient-vit-bench project), so the same deterministic 196/144/100/49 spatial
token subsets are applied to every image crop.

Text embeddings are computed once per distinct instruction string (FP32) and cached.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from model_wrapper import TokenReducedSiglipVision
from token_selection import build_configs

MODEL_ID = "google/siglip-base-patch16-224"
TEXT_MAX_LENGTH = 64   # SigLIP text tower was trained with padding="max_length", 64 tokens
DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_siglip(model_id: str, device: torch.device, attn_impl: str = "sdpa"):
    from transformers import SiglipModel
    return SiglipModel.from_pretrained(model_id, attn_implementation=attn_impl).float().eval().to(device)


def load_processor(model_id: str):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_id)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def score(image_embeds: torch.Tensor, text_norm: torch.Tensor, logit_scale: float, logit_bias: float) -> torch.Tensor:
    """SigLIP logits (n_images, n_texts) in FP32 regardless of the image-tower precision."""
    img = image_embeds.float()
    img = img / img.norm(p=2, dim=-1, keepdim=True)
    return img @ text_norm.t() * logit_scale + logit_bias


class SiglipScorer:
    """Scores image crops against instruction strings for a set of token configurations."""

    def __init__(self, model, processor, tokenizer, device: torch.device, dtype: torch.dtype = torch.float32):
        self.model, self.processor, self.tokenizer = model, processor, tokenizer
        self.device, self.dtype = device, dtype
        self.wrapper = TokenReducedSiglipVision(model)
        self.logit_scale = float(model.logit_scale.float().exp())
        self.logit_bias = float(model.logit_bias.float())
        self.configs = {k: v.to(device) for k, v in build_configs(14, (14, 12, 10, 7)).items()}
        self._text_cache: dict[str, torch.Tensor] = {}

    # -- text -----------------------------------------------------------------
    def tokenize(self, texts: list[str]) -> torch.Tensor:
        ids = self.tokenizer(texts, padding="max_length", max_length=TEXT_MAX_LENGTH,
                             truncation=True, return_tensors="pt")["input_ids"]
        assert ids.shape == (len(texts), TEXT_MAX_LENGTH), ids.shape
        return ids.to(self.device)

    @torch.inference_mode()
    def cache_texts(self, texts) -> None:
        """Compute normalised FP32 text embeddings once for every distinct string."""
        new = sorted(set(texts) - set(self._text_cache))
        if not new:
            return
        was = next(self.model.text_model.parameters()).dtype
        self.model.text_model.float()
        emb = self.model.text_model(input_ids=self.tokenize(new)).pooler_output.float()
        self.model.text_model.to(was)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        for t, e in zip(new, emb):
            self._text_cache[t] = e

    def text_matrix(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self._text_cache[t] for t in texts])

    # -- image ----------------------------------------------------------------
    def preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        """SigLIP image processor (CPU): resize to 224x224, rescale, normalise."""
        px = self.processor(images=crops, return_tensors="pt")["pixel_values"]
        assert px.ndim == 4 and px.shape[1:] == (3, 224, 224), px.shape
        return px

    @torch.inference_mode()
    def image_embeds(self, pixel_values: torch.Tensor, config: str) -> torch.Tensor:
        px = pixel_values.to(device=self.device, dtype=self.dtype)
        _, pooled = self.wrapper(px, self.configs[config])
        return pooled

    @torch.inference_mode()
    def pair_logits(self, pixel_values: torch.Tensor, texts: list[str], config: str, chunk: int = 256) -> torch.Tensor:
        """Logit of crop i against text i (one text per crop), shape (n,)."""
        assert pixel_values.shape[0] == len(texts)
        out = []
        for s in range(0, len(texts), chunk):
            emb = self.image_embeds(pixel_values[s:s + chunk], config)
            t = self.text_matrix(texts[s:s + chunk])
            img = emb.float() / emb.float().norm(dim=-1, keepdim=True)
            out.append((img * t).sum(-1) * self.logit_scale + self.logit_bias)
        return torch.cat(out) if out else torch.empty(0, device=self.device)


# -- validation -------------------------------------------------------------------
@torch.inference_mode()
def validate_against_siglip(scorer: SiglipScorer, pixel_values: torch.Tensor, texts: list[str],
                            atol: float = 1e-4, rtol: float = 1e-5) -> dict:
    """196-token custom scoring vs ``SiglipModel(input_ids, pixel_values).logits_per_image``.

    Must be called with the model and scorer in FP32. Compares the full
    (n_images x n_texts) logit matrix, the image and text embeddings, and the
    argmax per image, and the per-crop pair logits used by the agent.
    """
    model = scorer.model
    assert next(model.parameters()).dtype == torch.float32, "run validation in FP32"
    uniq = sorted(set(texts))
    scorer.cache_texts(uniq)
    px = pixel_values.to(scorer.device, torch.float32)
    ids = scorer.tokenize(uniq)
    ref = model(input_ids=ids, pixel_values=px)
    ref_logits = ref.logits_per_image.float()
    _, pooled = scorer.wrapper(px, torch.arange(scorer.wrapper.num_patches, device=scorer.device))
    tmat = scorer.text_matrix(uniq)
    ours = score(pooled, tmat, scorer.logit_scale, scorer.logit_bias)
    pair = scorer.pair_logits(px, texts, "196")
    col = torch.as_tensor([uniq.index(t) for t in texts], device=scorer.device)
    ref_pair = ref_logits.gather(1, col[:, None])[:, 0]
    img_norm = pooled.float() / pooled.float().norm(dim=-1, keepdim=True)
    res = {
        "n_images": int(px.shape[0]), "n_texts": len(uniq),
        "max_abs_diff_logits": float((ours - ref_logits).abs().max()),
        "max_abs_diff_pair_logits": float((pair - ref_pair).abs().max()),
        "max_abs_ref_logit": float(ref_logits.abs().max()),
        "max_abs_diff_image_embeds": float((img_norm - ref.image_embeds.float()).abs().max()),
        "max_abs_diff_text_embeds": float((tmat - ref.text_embeds.float()).abs().max()),
        "argmax_agreement": float((ours.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
        "atol": atol, "rtol": rtol,
    }
    res["passed"] = bool(torch.allclose(ours, ref_logits, atol=atol, rtol=rtol)
                         and torch.allclose(pair, ref_pair, atol=atol, rtol=rtol)
                         and res["argmax_agreement"] == 1.0)
    return res
