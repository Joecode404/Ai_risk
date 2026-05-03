from PIL import Image


def ensure_pil_image(image):
    """
    Make sure the input is a PIL image.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")