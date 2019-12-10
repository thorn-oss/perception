import cv2

from ..hasher import ImageHasher


class DHash(ImageHasher):
    """A hash based on the differences between adjacent pixels.
    Implementation based on that of
    `ImageHash <https://github.com/JohannesBuchner/imagehash>`_.
    """
    dtype = 'bool'
    distance_metric = 'hamming'

    def __init__(self, hash_size=8):
        assert hash_size > 1, 'Hash size must be greater than 1.'
        self.hash_size = hash_size
        self.hash_length = hash_size * hash_size

    def _compute(self, image):
        image = cv2.resize(
            image,
            dsize=(self.hash_size + 1, self.hash_size),
            interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        previous = image[:, :-1]
        current = image[:, 1:]
        difference = previous > current
        return difference.flatten()
