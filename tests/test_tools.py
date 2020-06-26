# pylint: disable=invalid-name
import tempfile
import shutil
import os

import numpy as np
import pytest

from perception import hashers, tools, testing


def test_deduplicate():
    directory = tempfile.TemporaryDirectory()
    original = testing.DEFAULT_TEST_IMAGES[0]
    duplicate = os.path.join(directory.name, 'image1.jpg')
    shutil.copy(original, duplicate)
    pairs = tools.deduplicate(
        files=[
            testing.DEFAULT_TEST_IMAGES[0], testing.DEFAULT_TEST_IMAGES[1],
            duplicate
        ],
        hashers=[(hashers.PHash(hash_size=16), 0.25)])
    assert len(pairs) == 1
    file1, file2 = pairs[0]
    assert ((file1 == duplicate) and
            (file2 == original)) or ((file1 == original) and
                                     (file2 == duplicate))


def test_deduplicate_u8():
    # This test verifies that extensions.compute_euclidean_pairwise_overlap
    # works properly.
    directory = tempfile.TemporaryDirectory()
    original = testing.DEFAULT_TEST_IMAGES[0]
    duplicate = os.path.join(directory.name, 'image1.jpg')
    shutil.copy(original, duplicate)
    pairs = tools.deduplicate(
        files=[
            testing.DEFAULT_TEST_IMAGES[0], testing.DEFAULT_TEST_IMAGES[1],
            duplicate
        ],
        hashers=[(hashers.PHashU8(hash_size=16), 10)])
    assert len(pairs) == 1
    file1, file2 = pairs[0]
    assert ((file1 == duplicate) and
            (file2 == original)) or ((file1 == original) and
                                     (file2 == duplicate))


def test_compute_euclidean_pairwise_overlap():
    # The purpose of this test is to verify that the handling of
    # deduplication with files that have multiple hashes works
    # properly. This is particularly important for video where
    # we are likely to have many hashes.
    X = np.array([
        # File 1
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        # File 2
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        # File 3
        [3, 3, 3],
        [4, 4, 4],
        # File 4
        [5, 5, 5],
        [6, 6, 6]
    ])

    # Use grouped files.
    counts = np.array([3, 3, 2, 2])
    expected = np.array([[2 / 3, 2 / 3], [0, 0], [0, 0], [1 / 3, 1 / 2],
                         [0, 0], [0, 0]])
    actual = tools.extensions.compute_euclidean_pairwise_overlap(
        X=X.astype('int32'), threshold=1, counts=counts.astype('int32'))
    assert (expected == actual).all()

    # Use ungrouped files.
    X = np.array([
        # File 1
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 1],
    ])
    expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0]])
    actual = tools.extensions.compute_euclidean_pairwise_overlap(
        X=X.astype('int32'), threshold=1)
    assert (expected == actual).all()


def test_api_is_over_https():
    matcher_https = tools.SaferMatcher(
        api_key='foo', url='https://www.example.com/')
    assert matcher_https

    if 'SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP' in os.environ:
        del os.environ['SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP']
    with pytest.raises(ValueError):
        tools.SaferMatcher(api_key='foo', url='http://www.example.com/')

    os.environ['SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP'] = '1'
    matcher_http_with_escape_hatch = tools.SaferMatcher(
        api_key='foo', url='http://www.example.com/')
    assert matcher_http_with_escape_hatch


def test_unletterbox():
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    padded = np.zeros((image.shape[0] + 100, image.shape[1] + 50, 3),
                      dtype='uint8')
    padded[50:50 + image.shape[0], 25:25 + image.shape[1]] = image
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded)
    assert y1 == 50
    assert y2 == 50 + image.shape[0]
    assert x1 == 25
    assert x2 == 25 + image.shape[1]


def test_unletterbox_noblackbars():
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(image)
    assert x1 == 0
    assert y1 == 0
    assert x2 == image.shape[1]
    assert y2 == image.shape[0]
