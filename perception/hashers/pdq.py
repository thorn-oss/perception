# pylint: disable=invalid-name
import pdqhash

from .hasher import Hasher


class PDQHash(Hasher):
    """The Facebook PDQ hash. Based on the original implementation located at
    the `official repository <https://github.com/facebook/ThreatExchange>`_.
    """
    distance_metric = 'hamming'
    dtype = 'bool'
    hash_length = 256

    def _compute(self, image):
        return pdqhash.compute(image)[0] > 0

    def _compute_with_quality(self, image):
        hash_vector, quality = pdqhash.compute(image)
        return hash_vector > 0, quality

    # pylint: disable=no-self-use
    def _compute_isometric(self, image):
        hash_vectors, _ = pdqhash.compute_dihedral(image)
        names = ['r0', 'r90', 'r180', 'r270', 'fv', 'fh', 'r90fv', 'r90fh']
        return dict(zip(names, hash_vectors))
