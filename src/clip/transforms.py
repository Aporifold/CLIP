import torchvision.transforms as tvf

__all__ = ["get_image_preprocess"]

# RGB statistics of the CLIP pretraining dataset (400M image-text pairs)
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert("RGB")


def get_image_preprocess(image_resolution: int = 224) -> tvf.Compose:
    """Build the image preprocessing pipeline for CLIP training/inference.

    Official implementation (based on torchvision):

    Resize(image_resolution, interpolation=BICUBIC)
    -> CenterCrop(image_resolution)
    -> Convert to RGB
    -> ToTensor
    -> Normalize(mean, std), with statistics of the CLIP pretraining dataset

    Args:
        image_resolution: Input image resolution
    Returns:
        A callable preprocessing module
    """
    return tvf.Compose(
        [
            tvf.Resize(image_resolution, interpolation=tvf.InterpolationMode.BICUBIC),
            tvf.CenterCrop(image_resolution),
            _convert_to_rgb,
            tvf.ToTensor(),
            tvf.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
