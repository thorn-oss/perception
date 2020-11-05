# pylint: disable=invalid-name
import pdqhash

from ..hasher import ImageHasher


class PDQHash(ImageHasher):
    """The Facebook PDQ hash. Based on the original implementation located at
    the `official repository <https://github.com/facebook/ThreatExchange>`_.
    """
    distance_metric = 'hamming'
    dtype = 'bool'
    hash_length = 256

    def _compute(self, image):
        """
        Compute image dataframe.

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        return pdqhash.compute(image)[0] > 0

    def _compute_with_quality(self, image):
        """
        Compute the vector of the image.

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        hash_vector, quality = pdqhash.compute(image)
        return hash_vector > 0, quality

    # pylint: disable=no-self-use
    def _compute_isometric(self, image):
        """
        Compute image isometric image.

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        hash_vectors, _ = pdqhash.compute_dihedral(image)
        names = ['r0', 'r90', 'r180', 'r270', 'fv', 'fh', 'r90fv', 'r90fh']
        return dict(zip(names, hash_vectors))


class PDQHashF(PDQHash):
    dtype = 'float32'
    distance_metric = 'euclidean'
    hash_length = 256

    # pylint: disable=no-self-use
    def _compute(self, image):
        """
        Compute the number of an image.

        Args:
            self: (todo): write your description
            image: (array): write your description
        """
        return pdqhash.compute_float(image)[0]
