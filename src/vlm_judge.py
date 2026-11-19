"""Module de chargement, inférence et évaluation pour les VLMs légers (BLIP et GIT).

Ce module centralise les fonctions réutilisables pour :
- Charger les modèles BLIP et GIT avec quantification (fp16, int8, int4, fp32).
- Générer des légendes à partir d'une image.
- Calculer la perte (loss) d'alignement image-légende.
- Juger de la cohérence et comparer deux légendes pour une même image.
"""

from __future__ import annotations

import math
import gc
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

MODEL_REGISTRY = {
    "blip": "Salesforce/blip-image-captioning-base",
    "git": "microsoft/git-base",
}


def get_default_device() -> str:
    """Retourne le device approprié ('cuda', 'mps' ou 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_quantization_config(precision: str) -> Optional[Any]:
    """Construit la configuration BitsAndBytesConfig selon le niveau demandé."""
    if precision in ("fp16", "fp32", "none", None):
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            f"La quantification {precision} nécessite la bibliothèque `bitsandbytes`."
        ) from exc

    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    raise ValueError(
        f"Précision inconnue : '{precision}'. Valeurs supportées : 'fp16', 'int8', 'int4', 'fp32'."
    )


def clear_device_cache() -> None:
    """Libère la mémoire GPU non utilisée."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_vlm(
    model_name: str = "blip",
    precision: str = "fp16",
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Charge le processeur et le modèle VLM spécifié.

    Args:
        model_name: 'blip' ou 'git'
        precision: 'fp16', 'int8', 'int4' ou 'fp32'
        device: 'cuda', 'cpu', ou None pour détection automatique

    Returns:
        Tuple (model, processor)
    """
    model_name_key = model_name.lower().strip()
    if model_name_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Modèle inconnu '{model_name}'. Modèles disponibles : {list(MODEL_REGISTRY.keys())}"
        )

    model_id = MODEL_REGISTRY[model_name_key]
    device = device or get_default_device()
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Si pas de GPU CUDA, int8 et int4 ne sont pas supportés par bitsandbytes
    effective_precision = precision
    if not is_cuda and precision in ("int8", "int4", "fp16"):
        effective_precision = "fp32"

    q_config = build_quantization_config(effective_precision) if is_cuda else None
    torch_dtype = torch.float16 if (is_cuda and effective_precision == "fp16") else torch.float32

    if model_name_key == "blip":
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(model_id)
        load_kwargs: Dict[str, Any] = {}
        if q_config is not None:
            load_kwargs["quantization_config"] = q_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            if is_cuda:
                load_kwargs["device_map"] = "auto"

        model = BlipForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        if not is_cuda or q_config is None and "device_map" not in load_kwargs:
            model = model.to(device)

    else:  # git
        from transformers import GitForCausalLM, GitProcessor

        processor = GitProcessor.from_pretrained(model_id)
        load_kwargs = {}
        if q_config is not None:
            load_kwargs["quantization_config"] = q_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            if is_cuda:
                load_kwargs["device_map"] = "auto"

        model = GitForCausalLM.from_pretrained(model_id, **load_kwargs)
        if not is_cuda or q_config is None and "device_map" not in load_kwargs:
            model = model.to(device)

    model.eval()
    return model, processor


def generate_caption(
    model: Any,
    processor: Any,
    image: Image.Image,
    model_name: str = "blip",
    max_new_tokens: int = 50,
) -> str:
    """Génère une légende pour une image donnée."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    device = getattr(model, "device", torch.device("cpu"))
    model_name_key = model_name.lower().strip()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if model_name_key == "git":
            output_ids = model.generate(
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=max_new_tokens,
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def compute_alignment_loss(
    model: Any,
    processor: Any,
    image: Image.Image,
    caption: str,
) -> float:
    """Calcule la perte (cross-entropy loss) entre l'image et une légende candidate."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    device = getattr(model, "device", torch.device("cpu"))
    inputs = processor(
        images=image,
        text=caption,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model(**inputs, labels=inputs["input_ids"])

    loss_val = float(output.loss.detach().cpu().item())
    return loss_val


def judge_caption(
    model: Any,
    processor: Any,
    image: Image.Image,
    caption: str,
    loss_threshold: float = 3.5,
) -> Dict[str, Any]:
    """Évalue la cohérence d'une légende vis-à-vis d'une image."""
    loss = compute_alignment_loss(model, processor, image, caption)
    score = -loss
    perplexity = math.exp(min(loss, 20.0))

    if loss <= 2.5:
        verdict = "Forte cohérence (légende très pertinente)"
        confidence = "Élevée"
    elif loss <= loss_threshold:
        verdict = "Cohérence modérée (légende plausible)"
        confidence = "Moyenne"
    else:
        verdict = "Incohérence probable (légende peu adaptée)"
        confidence = "Faible"

    return {
        "caption": caption,
        "loss": round(loss, 4),
        "score": round(score, 4),
        "perplexity": round(perplexity, 2),
        "verdict": verdict,
        "confidence": confidence,
    }


def compare_two_captions(
    model: Any,
    processor: Any,
    image: Image.Image,
    caption_a: str,
    caption_b: str,
) -> Dict[str, Any]:
    """Compare deux légendes pour une même image et sélectionne la meilleure (perte minimale)."""
    loss_a = compute_alignment_loss(model, processor, image, caption_a)
    loss_b = compute_alignment_loss(model, processor, image, caption_b)

    if loss_a < loss_b:
        winner = "Légende A"
        winner_text = caption_a
        delta = loss_b - loss_a
    elif loss_b < loss_a:
        winner = "Légende B"
        winner_text = caption_b
        delta = loss_a - loss_b
    else:
        winner = "Égalité"
        winner_text = "Les deux légendes ont la même perte"
        delta = 0.0

    return {
        "caption_a": caption_a,
        "loss_a": round(loss_a, 4),
        "score_a": round(-loss_a, 4),
        "caption_b": caption_b,
        "loss_b": round(loss_b, 4),
        "score_b": round(-loss_b, 4),
        "winner": winner,
        "winner_text": winner_text,
        "loss_difference": round(delta, 4),
    }
