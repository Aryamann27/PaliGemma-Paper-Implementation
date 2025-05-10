from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch
from process_images import process_images

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_propmpt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token*image_seq_len}{bos_token}{prefix_prompt}\n"

class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # These tokens are used for object detection (bouding boxes) 
        EXTRA_TOKENS += [
            F"<seg{i: 03d}" for i in range(128)
        ] # these tokens are used for object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_tokens = False
        tokenizer.add_eos_tokens = False

        self.tokenizer = tokenizer

    def __call__(
            self, text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Recieved {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factors = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )

        # convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        #convert the numpy array to pytorch tensor
        pixel_values = torch.tensor(pixel_values)

        # prepend a 'self.image_seq_length' number of image tokens to the prompt
        input_strings = [
            # create placeholder tokens for the text (or the prompt)
            add_image_tokens_to_propmpt(
            prefix_prompt=prompt,
            bos_token=self.tokenizer.bos_tokes,
            image_seq_len=self.image_seq_length,
            image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        # returns the input_ids and attention_mask as pytorch tensors
        inputs = self.tokenizer(
            input_strings, 
            return_tensors="pt",
            padding=padding, 
            truncation=truncation
        )
        return_data = {"pixel_values":pixel_values, **inputs}

        return return_data

