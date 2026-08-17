import torch
import torch.nn as nn

__all__ = ["ClipLoss", "clip_loss"]


class ClipLoss(nn.Module):
    """CLIP contrastive learning loss (Symmetric InfoNCE loss).

    Core Idea: Pull paired (image, text) features closer together while
        pushing unpaired samples apart.

    Workflow:
    1. Apply L2 normalization to `image_features` and `text_features`.
    2. Calculate image-text similarity matrix: `logits = (I @ T.T) * logit_scale.exp()`, with shape `[N, N]`.
    3. Image-to-text direction: `loss_i = CrossEntropy(logits, torch.arange(N))`
    4. Text-to-image direction: `loss_t = CrossEntropy(logits.T, torch.arange(N))`
    5. Return average loss of two direction.
    """

    def __init__(self, gather_with_grad: bool = False, local_loss: bool = False):
        super(ClipLoss, self).__init__()
        self.gather_with_grad = gather_with_grad
        self.local_loss = local_loss
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate symmetric contrastive loss.

        Args:
            image_features (torch.Tensor): Unnormalized Image features, of shape (N, D)
            text_features (torch.Tensor):  Unnormalized Text features, of shape (N, D)
            logit_scale (torch.Tensor):    Temperature on a log scale.
        Returns:
            torch.Tensor: Scalar loss
        """
        N = image_features.size(0)

        # 1. Normalize image and text features
        image_features = image_features / image_features.norm(p=2, keepdim=True)
        text_features = text_features / text_features.norm(p=2, keepdim=True)

        # 2. Calculate cosine similarity matrix
        logits = (image_features @ text_features.t()) * logit_scale.exp()  # (N, N)

        # 3. Symmetric InfoNCE loss
        labels = torch.arange(N)
        loss_i = self.loss(logits, labels)
        loss_t = self.loss(logits.t(), labels)
        loss = (loss_i + loss_t) / 2
        return loss


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Functional interface for `CLIPLoss.forward(...)`."""
    return ClipLoss()(image_features, text_features, logit_scale)
