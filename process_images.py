from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

def resize(
        image: Image,
        size: Tuple[int, int],
        resample: Image.Resampling=None,
        reducing_gap: Optional[int]=None
)-> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (height, width), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
        image: np.ndarray, scale: float, dtype: np.dtype=np.float32
)-> np.ndarray:
    rescaled_image = image*scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
)-> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

    
def process_images(
    images: List[Image.Image],
    size: Dict[str, int]=None,
    resample: Image.Resampling=None,
    rescale_factor: float=None,
    image_mean: Optional[Union[float, List[float]]]=None,
    image_std: Optional[Union[float, List[float]]]=None,
    ) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # convert each image to numpy array
    images = [np.array(image) for image in images]

    # rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]

    # normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    #m move the channel dimension to the first dimenstio. the model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2,0,1) for image in images]
    return images