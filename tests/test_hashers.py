# pylint: disable=invalid-name

import os
import pytest

from perception import hashers, testing

TEST_IMAGES = [
    os.path.join('tests', 'images', f'image{n}.jpg') for n in range(1, 11)
]


# The PDQ hash isometric computation is inexact. See
# https://github.com/faustomorales/pdqhash-python/blob/master/tests/test_compute.py
# for details.
@pytest.mark.parametrize(
    "hasher_class,pil_opencv_threshold,transform_threshold,opencv_hasher",
    [(hashers.AverageHash, 0.1, 0.1, False),
     (hashers.WaveletHash, 0.1, 0.1, False), (hashers.PHash, 0.1, 0.1, False),
     (hashers.PDQHash, 0.1, 0.15, False), (hashers.DHash, 0.1, 0.1, False),
     (hashers.MarrHildreth, 0.1, 0.1, True),
     (hashers.BlockMean, 0.1, 0.1, True),
     (hashers.ColorMoment, 10, 0.1, True)])
def test_hashing_common(hasher_class, pil_opencv_threshold,
                        transform_threshold, opencv_hasher):
    testing.test_hasher_integrity(
        hasher=hasher_class(),
        pil_opencv_threshold=pil_opencv_threshold,
        transform_threshold=transform_threshold,
        opencv_hasher=opencv_hasher)
