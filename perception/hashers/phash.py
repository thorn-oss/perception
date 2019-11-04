import scipy.fftpack
import numpy as np
import cv2

from .hasher import Hasher
from . import tools


class PHash(Hasher):
    """Also known as the DCT hash, a hash based on discrete cosine transforms of images.
    See `complete paper <https://www.phash.org/docs/pubs/thesis_zauner.pdf>`_ for
    details. Implementation based on that of
    `ImageHash <https://github.com/JohannesBuchner/imagehash>`_.

    Args:
        hash_size: The number of DCT elements to retain (the hash length
            will be hash_size * hash_size).
        highfreq_factor: The multiple of the hash size to resize the input
            image to before computing the DCT.
        exclude_first_term: WHether to exclude the first term of the DCT
    """
    distance_metric = 'hamming'
    dtype = 'bool'

    def __init__(self,
                 hash_size=8,
                 highfreq_factor=4,
                 exclude_first_term=False):
        assert hash_size >= 2, 'Hash size must be greater than or equal to 2'
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.exclude_first_term = exclude_first_term
        self.hash_length = hash_size * hash_size
        if exclude_first_term:
            self.hash_length -= 1

    def _compute_dct(self, image):
        img_size = self.hash_size * self.highfreq_factor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(
            image, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
        dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
        return dct[:self.hash_size, :self.hash_size]

    # pylint: disable=no-self-use
    def _dct_to_hash(self, dct):
        dct = dct.flatten()
        if self.exclude_first_term:
            dct = dct[1:]
        return dct > np.median(dct)

    def _compute(self, image):
        dct = self._compute_dct(image)
        return self._dct_to_hash(dct)

    def _compute_isometric(self, image):
        return {
            transform_name: self._dct_to_hash(dct)
            for transform_name, dct in tools.get_isometric_dct_transforms(
                self._compute_dct(image)).items()
        }
