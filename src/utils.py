def to_rgb(img):
    from torch import Tensor

    if isinstance(img, Tensor):
        return img.repeat(3, 1, 1)
    return img.convert("RGB")


def ToRGB():
    from torchvision import transforms as T

    return T.Lambda(to_rgb)
