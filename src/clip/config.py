from dataclasses import dataclass, field

__all__ = ["VisionConfig", "TextConfig", "CLIPConfig"]


@dataclass
class VisionConfig:
    """Image encoder configuration.

    Attributes:
        image_resolution: Input image resolution (side length of a square image)
        vision_layers: Number of transformer layers; use a 4-tuple of layer counts
            (e.g. ``(3, 4, 6, 3)``) for the ResNet architecture
        vision_width: Hidden dimension
        vision_patch_size: Patch size for the ViT architecture
        vision_heads: Number of attention heads
    """

    image_resolution: int = 224
    vision_layers: int = 12
    vision_width: int = 768
    vision_patch_size: int = 32
    vision_heads: int = 12


@dataclass
class TextConfig:
    """Text encoder configuration.

    Attributes:
        context_length: Maximum text sequence length
        vocab_size: BPE vocabulary size (49408 in the official implementation)
        transformer_width: Hidden dimension
        transformer_heads: Number of attention heads
        transformer_layers: Number of transformer layers
    """

    context_length: int = 77
    vocab_size: int = 49408
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12


@dataclass
class CLIPConfig:
    """Overall CLIP model configuration.

    Attributes:
        embed_dim: Shared embedding dimension for both image/text projections
        logit_scale_init: Initial value of the temperature parameter (1/0.07 officially)
        vision: Image encoder configuration
        text: Text encoder configuration
    """

    embed_dim: int = 512
    logit_scale_init: float = 1 / 0.07
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
