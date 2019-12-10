import cv2

from ..hasher import ImageHasher
from .. import tools


class AverageHash(ImageHasher):
    """Computes a simple hash comparing the intensity of each
    pixel in a resized version of the image to the mean.
    Implementation based on that of
    `ImageHash <https://github.com/JohannesBuchner/imagehash>`_."""
    distance_metric = 'hamming'
    dtype = 'bool'

    def __init__(self, hash_size=8):
        assert hash_size >= 2, "Hash size must be greater than or equal to 2."
        self.hash_size = hash_size
        self.hash_length = hash_size * hash_size

    def _compute(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(
            image,
            dsize=(self.hash_size, self.hash_size),
            interpolation=cv2.INTER_AREA)
        diff = image > image.mean()
        return diff.flatten()

    def _compute_isometric_from_hash(self, vector):
        return {
            transform_name: diff.flatten()
            for transform_name, diff in tools.get_isometric_transforms(
                vector.reshape(self.hash_size, self.hash_size, 1),
                require_color=False).items()
        }
