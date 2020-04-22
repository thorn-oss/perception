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
    image = hashers.tools.read('perception/testing/images/image1.jpg')
    padded = np.zeros((image.shape[0] + 100, image.shape[1] + 50, 3),
                      dtype='uint8')
    padded[50:50 + image.shape[0], 25:25 + image.shape[1]] = image
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded)
    assert y1 == 50
    assert y2 == 50 + image.shape[0]
    assert x1 == 25
    assert x2 == 25 + image.shape[1]
