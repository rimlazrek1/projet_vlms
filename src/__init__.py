"""Package src pour le projet de jugement image-légende avec VLMs légers."""

from src.vlm_judge import (
    MODEL_REGISTRY,
    build_quantization_config,
    clear_device_cache,
    compare_two_captions,
    compute_alignment_loss,
    generate_caption,
    judge_caption,
    load_vlm,
)

__all__ = [
    "MODEL_REGISTRY",
    "build_quantization_config",
    "clear_device_cache",
    "compare_two_captions",
    "compute_alignment_loss",
    "generate_caption",
    "judge_caption",
    "load_vlm",
]
