# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image preprocessing for Moondream2's overlapping global/local crop encoder."""

import math
from typing import Optional

import numpy as np
from PIL import Image

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, make_list_of_images, to_numpy_array

__all__ = ["Moondream2ImageProcessor"]


def _select_tiling(height, width, crop_size, max_crops, overlap_margin, patch_size):
    """Select an aspect-ratio preserving local-tile grid, capped at max_crops."""
    usable = crop_size - 2 * overlap_margin * patch_size
    ratio = width / height
    best = (1, 1)
    best_score = float("inf")
    for tiles_h in range(1, max_crops + 1):
        for tiles_w in range(1, max_crops + 1):
            if tiles_h * tiles_w > max_crops:
                continue
            score = abs(math.log((tiles_w / tiles_h) / ratio))
            # Prefer the largest matching layout for better local-image detail.
            score -= (tiles_h * tiles_w) * 1e-4
            if score < best_score:
                best_score, best = score, (tiles_h, tiles_w)
    return best, usable


class Moondream2ImageProcessor(BaseImageProcessor):
    """Create the global crop plus overlapping local crops expected by Moondream2.

    ``pixel_values`` is padded to the largest crop count in a batch.  The
    companion ``crop_counts`` and ``image_sizes`` fields preserve which local
    crops are valid and their `(tiles_h, tiles_w)` reconstruction layout.
    """

    model_input_names = ["pixel_values", "image_sizes", "crop_counts"]

    def __init__(
        self,
        crop_size=378,
        patch_size=14,
        max_crops=12,
        overlap_margin=4,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_convert_rgb=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.size = {"height": crop_size, "width": crop_size}

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB") if self.do_convert_rgb else image
        array = to_numpy_array(image)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = array.transpose([1, 2, 0])
        if array.dtype != np.uint8:
            array = np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB") if self.do_convert_rgb else Image.fromarray(array)

    def _make_crops(self, image):
        width, height = image.size
        (tiles_h, tiles_w), usable = _select_tiling(
            height, width, self.crop_size, self.max_crops, self.overlap_margin, self.patch_size
        )
        global_crop = image.resize((self.crop_size, self.crop_size), Image.Resampling.LANCZOS)
        local_width = tiles_w * usable + 2 * self.overlap_margin * self.patch_size
        local_height = tiles_h * usable + 2 * self.overlap_margin * self.patch_size
        local_image = image.resize((local_width, local_height), Image.Resampling.LANCZOS)
        crops = [global_crop]
        for row in range(tiles_h):
            for col in range(tiles_w):
                left, top = col * usable, row * usable
                crops.append(local_image.crop((left, top, left + self.crop_size, top + self.crop_size)))
        return crops, (tiles_h, tiles_w)

    def _normalize(self, crop, do_rescale, do_normalize):
        values = np.asarray(crop).astype(np.float32)
        if do_rescale:
            values *= self.rescale_factor
        if do_normalize:
            values = (values - np.asarray(self.image_mean, dtype=np.float32)) / np.asarray(self.image_std, dtype=np.float32)
        return values.transpose([2, 0, 1])

    def preprocess(
        self,
        images: ImageInput,
        do_rescale: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None:
            raise ValueError("images must be specified")
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_list = make_list_of_images(images)
        batched_crops, image_sizes = [], []
        for image in image_list:
            crops, layout = self._make_crops(self._to_pil(image))
            batched_crops.append([self._normalize(crop, do_rescale, do_normalize) for crop in crops])
            image_sizes.append(layout)
        max_count = max(len(crops) for crops in batched_crops)
        pixel_values = np.zeros(
            [len(batched_crops), max_count, 3, self.crop_size, self.crop_size], dtype=np.float32
        )
        crop_counts = []
        for index, crops in enumerate(batched_crops):
            pixel_values[index, : len(crops)] = np.asarray(crops)
            crop_counts.append(len(crops))
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_sizes": np.asarray(image_sizes, dtype=np.int64),
                "crop_counts": np.asarray(crop_counts, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )
