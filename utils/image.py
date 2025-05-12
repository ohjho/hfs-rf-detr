from typing import Tuple, Union
from PIL import Image
import numpy as np

def calculate_resolution_wh(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:

    if isinstance(image, Image.Image):
        return image.size
    elif isinstance(image, np.ndarray):
        if image.ndim >= 2:
            h, w = image.shape[:2]
            return w, h
        else:
            raise ValueError("Input numpy array image must have at least 2 dimensions (height, width).")
    else:
        raise TypeError("Input image must be a Pillow Image or a numpy array.")
