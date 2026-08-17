import math

import torch
import torch.nn as nn

from .config import CLIPConfig, TextConfig, VisionConfig

__all__ = ["ImageEncoder", "TextEncoder", "CLIP"]


class ImageEncoder(nn.Module):
    """Image encoder that reuses ``transformers.CLIPVisionModelWithProjection``.

    Takes images of shape [B, 3, H, W] and returns features after the projection
    layer, of shape [B, embed_dim] (unnormalized; normalization happens in
    ``CLIP.forward`` / the loss).

    Args:
        config: Image encoder configuration
        output_dim: Projection output dimension (defaults to ``vision_width``,
            i.e. no dimensionality reduction)
        from_pretrained: When set, loads the architecture and weights from a
            transformers pretrained model; ``config`` is then overridden by the
            checkpoint's configuration
    """

    def __init__(
        self,
        config: VisionConfig,
        output_dim: int | None = None,
        from_pretrained: str | None = None,
    ):
        super().__init__()
        self.config = config

        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        if from_pretrained is not None:
            # Architecture + weights both come from the pretrained checkpoint
            self.model = CLIPVisionModelWithProjection.from_pretrained(from_pretrained)
        else:
            hf_config = CLIPVisionConfig(
                hidden_size=config.vision_width,
                num_hidden_layers=config.vision_layers,
                num_attention_heads=config.vision_heads,
                image_size=config.image_resolution,
                patch_size=config.vision_patch_size,
                projection_dim=output_dim or config.vision_width,
            )
            self.model = CLIPVisionModelWithProjection(hf_config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images into feature vectors of shape [B, embed_dim]."""
        return self.model(image).image_embeds


class TextEncoder(nn.Module):
    """Text encoder that reuses ``transformers.CLIPTextModelWithProjection``.

    Takes token id sequences of shape (B, context_length) and returns features
    after the projection layer, of shape (B, embed_dim) (unnormalized). EOT
    feature extraction is handled internally by transformers.

    Args:
        config: Text encoder configuration
        output_dim: Projection output dimension (defaults to
            ``transformer_width``, i.e. no dimensionality reduction)
        from_pretrained: When set, loads the architecture and weights from a
            transformers pretrained model; ``config`` is then overridden by the
            checkpoint's configuration
    """

    def __init__(
        self,
        config: TextConfig,
        output_dim: int | None = None,
        from_pretrained: str | None = None,
    ):
        super().__init__()
        self.config = config

        from transformers import CLIPTextConfig, CLIPTextModelWithProjection

        if from_pretrained is not None:
            self.model = CLIPTextModelWithProjection.from_pretrained(from_pretrained)
        else:
            hf_config = CLIPTextConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.transformer_width,
                num_hidden_layers=config.transformer_layers,
                num_attention_heads=config.transformer_heads,
                max_position_embeddings=config.context_length,
                projection_dim=output_dim or config.transformer_width,
            )
            self.model = CLIPTextModelWithProjection(hf_config)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Encode texts into feature vectors of shape [B, embed_dim]."""
        return self.model(text).text_embeds


class CLIP(nn.Module):
    """CLIP multimodal contrastive learning model.

    Structure:
    - image_encoder: Image encoder (transformers)
    - text_encoder:  Text encoder (transformers)
    - logit_scale:   Learnable temperature parameter, initialized to log(1/0.07)

    During training the contrastive loss is computed via
    :class:`clip.loss.ClipLoss`; at inference, zero-shot image classification can
    be performed based on text descriptions.

    Args:
        config: Model configuration (only a placeholder when using
            ``from_pretrained``; the actual configuration comes from the
            checkpoint)
        from_pretrained: When set, loads the full weights from a transformers
            pretrained model, e.g. ``"openai/clip-vit-base-patch32"``
    """

    def __init__(self, config: CLIPConfig, from_pretrained: str | None = None):
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(
            config.vision,
            output_dim=config.embed_dim,
            from_pretrained=from_pretrained,
        )
        self.text_encoder = TextEncoder(
            config.text,
            output_dim=config.embed_dim,
            from_pretrained=from_pretrained,
        )

        if from_pretrained is not None:
            try:
                from transformers import CLIPModel

                # The weight files were cached by the from_pretrained calls above;
                # here we only read the initial temperature value
                logit = CLIPModel.from_pretrained(from_pretrained).logit_scale.item()
            except Exception:
                logit = math.log(config.logit_scale_init)
        else:
            logit = math.log(config.logit_scale_init)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit)

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> "CLIP":
        """Load a complete CLIP from a transformers pretrained model."""
        return cls(CLIPConfig(), from_pretrained=pretrained_model)

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating point precision currently used by the model
        (for fp16/bf16 support)."""
        return self.image_encoder.model.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images into feature vectors of shape [B, embed_dim]."""
        return self.image_encoder(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode texts into feature vectors of shape [B, embed_dim]."""
        return self.text_encoder(text)

    def forward(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute image-text contrastive logits.

        Args:
            image: Image tensor, shape (B, 3, H, W)
            text:  Token sequence, shape (B, context_length)
        Returns:
            logits_per_image: Shape (B, B), entry (i, j) is the similarity between
                `I_i` and `T_j`
            logits_per_text:  (B, B), transpose of `logits_per_image`
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
