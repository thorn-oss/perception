import cv2
import numpy as np

import pywt

from ..hasher import ImageHasher


class WaveletHash(ImageHasher):
    """Similar to PHash but using wavelets instead of DCT.
    Implementation based on that of
    `ImageHash <https://github.com/JohannesBuchner/imagehash>`_.
    """
    distance_metric = 'hamming'
    dtype = 'bool'

    def __init__(self, hash_size=8, image_scale=None, mode='haar'):
        assert hash_size & (
            hash_size - 1) == 0, "Hash size must be a power of 2."
        if image_scale is not None:
            assert (
                image_scale &
                (image_scale - 1) == 0), "Image scale must be a power of 2."
            assert image_scale >= hash_size, \
                "Image scale must be greater than or equal to than hash size."
        self.hash_size = hash_size
        self.image_scale = image_scale
        self.mode = mode
        self.hash_length = hash_size * hash_size

    def _compute(self, image):
        if self.image_scale is None:
            image_scale = max(2**int(np.log2(min(image.shape[:2]))),
                              self.hash_size)
        else:
            image_scale = self.image_scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(
            image,
            dsize=(image_scale, image_scale),
            interpolation=cv2.INTER_AREA)
        image = np.float32(image) / 255

        ll_max_level = int(np.log2(image_scale))
        level = int(np.log2(self.hash_size))
        dwt_level = ll_max_level - level

        if self.mode == 'haar':
            coeffs = pywt.wavedec2(image, 'haar', level=ll_max_level)
            coeffs = list(coeffs)
            coeffs[0] *= 0
            image = pywt.waverec2(coeffs, 'haar')

        coeffs = pywt.wavedec2(image, self.mode, level=dwt_level)
        dwt_low = coeffs[0]

        # Subtract median and compute hash
        med = np.median(dwt_low)
        diff = dwt_low > med

        return diff.flatten()
