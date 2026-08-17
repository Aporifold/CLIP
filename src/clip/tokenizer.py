import torch
import torch.nn as nn

__all__ = ["Tokenizer"]


class Tokenizer(nn.Module):
    """CLIP Text Tokenizer.

    Reuse `CLIPTokenizer` (the GPT-2-style BPE from OpenAI's CLIP),
    eliminating the need to implement custom vocabulary loading and merge.
    By default, it loads the tokenizer for `openai/clip-vit-base-patch32`
    from huggingface hub (vocab size: 49,408).
    """

    def __init__(
        self,
        pretrained_model: str = "openai/clip-vit-base-patch32",
        tokenizer=None,
    ):
        super(Tokenizer, self).__init__()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import CLIPTokenizer, CLIPTokenizerFast

            try:
                self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model)
            except Exception:
                self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)

        self._context_length = self.tokenizer.model_max_length or 77

    @property
    def eot_token_id(self) -> int:
        """Token id for `<|endoftext|>`."""
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

    @property
    def sot_token_id(self) -> int:
        """Token id for `<|startoftext|>`"""
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)

    @property
    def context_length(self) -> int:
        """Max sequence length of text."""
        return self._context_length

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.tokenizer)

    def tokenize(
        self,
        texts: str | list[str],
        context_length: int | None = None,
        truncate: bool = False,
    ) -> torch.Tensor:
        """Encode text into a tensor of token ids.

        Args:
            texts: A text string or a list of text strings.
            context_length: Target sequence length. Defaults to `self.context_length`.
            truncate: Whether to truncate sequences exceeding the `context_length`,
                otherwise raises an error.
        Returns:
            torch.Tensor: Token ids, of shape (B, context_length),
                excess part is padded with `<|endoftext|>` (pad token).
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            max_length=context_length,
            truncation=truncate,
            return_tensors="pt",
        )
        return encodings["input_ids"]

    def decode(self, token_ids) -> list[str]:
        """Decode the token id sequence back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
