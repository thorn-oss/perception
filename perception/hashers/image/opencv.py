import cv2
import numpy as np

from ..hasher import ImageHasher


class OpenCVHasher(ImageHasher):
    allow_parallel = False

    def __init__(self):
        if not hasattr(cv2, "img_hash"):
            raise RuntimeError(
                "You do not appear to have opencv-contrib installed. It is required for pure OpenCV hashers."
            )


class MarrHildreth(OpenCVHasher):
    """A wrapper around OpenCV's Marr-Hildreth hash.
    See `paper <https://www.phash.org/docs/pubs/thesis_zauner.pdf>`_ for details."""

    dtype = "bool"
    distance_metric = "hamming"
    hash_length = 576

    def __init__(self):
        super().__init__()
        self.hasher = cv2.img_hash.MarrHildrethHash.create()

    def _compute(self, image):
        return np.unpackbits(self.hasher.compute(image)[0])


class ColorMoment(OpenCVHasher):
    """A wrapper around OpenCV's Color Moments hash.
    See `paper <https://www.phash.org/docs/pubs/thesis_zauner.pdf>`_ for details."""

    dtype = "float32"
    distance_metric = "euclidean"
    hash_length = 42

    def __init__(self):
        super().__init__()
        self.hasher = cv2.img_hash.ColorMomentHash.create()

    def _compute(self, image):
        return 10000 * self.hasher.compute(image)[0]


class BlockMean(OpenCVHasher):
    """A wrapper around OpenCV's Block Mean hash.
    See `paper <https://www.phash.org/docs/pubs/thesis_zauner.pdf>`_ for details."""

    dtype = "bool"
    distance_metric = "hamming"
    hash_length = 968

    def __init__(self):
        super().__init__()
        self.hasher = cv2.img_hash.BlockMeanHash.create(1)

    def _compute(self, image):
        # https://stackoverflow.com/questions/54762896/why-cv2-norm-hamming-gives-different-value-than-actual-hamming-distance
        return np.unpackbits(self.hasher.compute(image)[0])
