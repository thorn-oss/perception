import scipy.fftpack
import numpy as np
import cv2

from ..hasher import ImageHasher
from .. import tools


class PHash(ImageHasher):
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
        freq_shift: The number of DCT low frequency elements to skip.
    """
    distance_metric = 'hamming'
    dtype = 'bool'

    def __init__(self,
                 hash_size=8,
                 highfreq_factor=4,
                 exclude_first_term=False,
                 freq_shift=0):
        assert hash_size >= 2, 'Hash size must be greater than or equal to 2'
        assert freq_shift <= highfreq_factor * hash_size - hash_size, \
            'Frequency shift is too large for this hash size / highfreq_factor combination.'
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.exclude_first_term = exclude_first_term
        self.hash_length = hash_size * hash_size
        self.freq_shift = freq_shift
        if exclude_first_term:
            self.hash_length -= 1

    def _compute_dct(self, image):
        img_size = self.hash_size * self.highfreq_factor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(
            image, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
        dct = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0), axis=1)
        return dct[self.freq_shift:self.hash_size + self.freq_shift, self.
                   freq_shift:self.hash_size + self.freq_shift]

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


class PHashF(PHash):
    """A real-valued version of PHash. It
    returns the raw 32-bit floats in the DCT.
    For a more compact approach, see PHashU8."""
    dtype = 'float32'
    distance_metric = 'euclidean'

    def _dct_to_hash(self, dct):
        dct = dct.flatten()
        if self.exclude_first_term:
            dct = dct[1:]
        if (dct == 0).all():
            return None
        return dct


class PHashU8(PHash):
    """A real-valued version of PHash. It
    uses minimum / maximum scaling to convert
    DCT values to unsigned 8-bit integers (more
    compact than the 32-bit floats used by PHashF at
    the cost of precision)."""
    dtype = 'uint8'
    distance_metric = 'euclidean'

    def _dct_to_hash(self, dct):
        dct = dct.flatten()
        if self.exclude_first_term:
            dct = dct[1:]
        if (dct == 0).all():
            return None
        min_value = dct.min()
        max_value = dct.max()
        dct = np.uint8(255 * (dct - min_value) / (max_value - min_value))
        return dct
