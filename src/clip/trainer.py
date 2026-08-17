import torch
from torch.utils.data import DataLoader

from .loss import ClipLoss
from .model import CLIP

__all__ = ["Trainer"]


class Trainer:
    """Trainer framework for CLIP contrastive pretraining.

    Typical training loop:
    for epoch in range(num_epochs):
        for images, texts in dataloader:
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = loss_fn(image_features, text_features, model.logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model: CLIP,
        train_loader: DataLoader,
        loss_fn: ClipLoss | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn or ClipLoss()
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=5e-4)
        self.device = device

    def train_one_epoch(self) -> float:
        """Train for one epoch and return the average loss."""
        total_loss = 0.0
        for images, texts in self.train_loader:
            # 1. Encode images and texts
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            # 2. Compute symmetric InfoNCE loss
            loss = self.loss_fn(image_features, text_features, self.model.logit_scale)
            total_loss += loss.detach().float().item()
            # 3. Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.train_loader)

    def train(self, num_epochs: int) -> list[float]:
        """Train for `num_epochs` epochs, returning the average loss per epoch."""
        avg_losses = []
        for epoch in range(num_epochs):
            avg_loss = self.train_one_epoch()
            print(f"Epoch: [{epoch + 1}/{num_epochs}], Avg. Loss = {avg_loss}")
            avg_losses.append(avg_loss)
        return avg_losses
