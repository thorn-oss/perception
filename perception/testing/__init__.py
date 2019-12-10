# pylint: disable=invalid-name,too-many-locals
import os
import math
import typing
import pkg_resources

import cv2
import pytest
import numpy as np
import pandas as pd
from PIL import Image  # pylint: disable=import-error

from .. import hashers, tools

SIZES = {'float32': 32, 'uint8': 8, 'bool': 1}


def get_low_detail_image():
    v = np.arange(0, 50, 1)
    v = np.concatenate([v, v[::-1]])[np.newaxis, ]
    image = np.matmul(v.T, v)
    image = (image * 255 / image.max()).astype('uint8')
    image = image[..., np.newaxis].repeat(repeats=3, axis=2)
    image[:, 50:] = 0
    image[50:] = 0
    return image


LOW_DETAIL_IMAGE = get_low_detail_image()

DEFAULT_TEST_IMAGES = [
    pkg_resources.resource_filename(
        'perception', os.path.join('testing', 'images', f'image{n}.jpg'))
    for n in range(1, 11)
]
DEFAULT_TEST_VIDEOS = [
    pkg_resources.resource_filename(
        'perception', os.path.join('testing', 'videos', f'v{n}.m4v'))
    for n in range(1, 3)
]


@typing.no_type_check
def test_opencv_hasher(hasher: hashers.ImageHasher, image1: str, image2: str):
    # For OpenCV hashers we make sure the distance we compute
    # is the same as inside OpenCV
    f1 = image1
    f2 = image2
    opencv_distance = hasher.hasher.compare(
        hasher.hasher.compute(hashers.tools.read(f1)),
        hasher.hasher.compute(hashers.tools.read(f2)))
    if hasher.distance_metric == 'hamming':
        opencv_distance /= hasher.hash_length
    np.testing.assert_approx_equal(
        opencv_distance,
        hasher.compute_distance(hasher.compute(f1), hasher.compute(f2)),
        significant=4)


# pylint: disable=protected-access
def hash_dicts_to_df(hash_dicts, returns_multiple):
    assert all(
        h['error'] is None
        for h in hash_dicts), 'An error was found in the hash dictionaries'
    if returns_multiple:
        return pd.DataFrame({
            'filepath':
            tools.flatten(
                [[h['filepath']] * len(h['hash']) for h in hash_dicts]),
            'hash':
            tools.flatten([h['hash'] for h in hash_dicts])
        }).assign(error=None)
    return pd.DataFrame.from_records(hash_dicts).assign(error=None)


def test_hasher_parallelization(hasher, test_filepaths):
    filepaths_10x = test_filepaths * 10
    if not hasher.allow_parallel:
        with pytest.warns(UserWarning, match='cannot be used in parallel'):
            hashes_parallel_dicts = hasher.compute_parallel(
                filepaths=filepaths_10x)
    else:
        hashes_parallel_dicts = hasher.compute_parallel(
            filepaths=filepaths_10x)
    hashes_sequential_dicts = [{
        'filepath': filepath,
        'hash': hasher.compute(filepath),
        'error': None
    } for filepath in filepaths_10x]
    hashes_parallel = hash_dicts_to_df(
        hashes_parallel_dicts,
        returns_multiple=hasher.returns_multiple).sort_values(
            ['filepath', 'hash'])
    hashes_sequential = hash_dicts_to_df(
        hashes_sequential_dicts,
        returns_multiple=hasher.returns_multiple).sort_values(
            ['filepath', 'hash'])
    assert (hashes_sequential.hash.values == hashes_parallel.hash.values).all()
    assert (hashes_sequential.filepath.values == hashes_parallel.filepath.
            values).all()


def test_video_hasher_integrity(hasher: hashers.VideoHasher,
                                test_videos: typing.List[str] = None):
    if test_videos is None:
        test_videos = DEFAULT_TEST_VIDEOS
    test_hasher_parallelization(hasher, test_videos)


def test_image_hasher_integrity(hasher: hashers.ImageHasher,
                                pil_opencv_threshold: float,
                                transform_threshold: float,
                                test_images: typing.List[str] = None,
                                opencv_hasher: bool = False):
    """Test to ensure a hasher works correctly.

    Args:
        hasher: The hasher to test.
        test_images: A list of filepaths to images to use for testing.
        pil_opencv_threshold: The hash distance permitted for an image
            when loaded with OpenCV vs. PIL.
        transform_threshold: The permitted error in isometric transform
            hashes.
        opencv_hasher: Whether the hasher is an OpenCV hasher. Used to
            determine whether to check for consistent distances.
    """
    if test_images is None:
        test_images = DEFAULT_TEST_IMAGES
    assert len(test_images) >= 2, 'You must provide at least two test images.'
    image1 = test_images[0]
    image2 = test_images[1]
    hash1_1 = hasher.compute(image1)
    hash1_2 = hasher.compute(Image.open(image1))
    hash1_3 = hasher.compute(
        cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2RGB))

    hash2_1 = hasher.compute(image2)

    # There is a small distance because PIL and OpenCV read
    # JPEG images a little differently (e.g., libjpeg-turbo vs. libjpeg)
    assert hasher.compute_distance(hash1_1, hash1_2) < pil_opencv_threshold
    assert hasher.compute_distance(hash1_1, hash2_1) > pil_opencv_threshold
    assert hasher.compute_distance(hash1_1, hash1_3) == 0

    # Ensure the conversion to and from vectors works for both base64 and hex.
    assert hasher.vector_to_string(hasher.string_to_vector(hash2_1)) == hash2_1
    assert hasher.vector_to_string(
        hasher.string_to_vector(
            hasher.vector_to_string(
                hasher.string_to_vector(hash2_1), hash_format='hex'),
            hash_format='hex')) == hash2_1

    # Ensure parallelization works properly.
    test_hasher_parallelization(hasher=hasher, test_filepaths=test_images)

    # Ensure the isometric hashes computation work properly
    for image in test_images:
        transforms = hashers.tools.get_isometric_transforms(image)
        hashes_exp = {
            key: hasher.compute(value)
            for key, value in transforms.items()
        }
        hashes_act = hasher.compute_isometric(transforms['r0'])
        for transform_name in hashes_exp.keys():
            assert hasher.compute_distance(
                hashes_exp[transform_name],
                hashes_act[transform_name]) < transform_threshold

    # Verify that hashes are the correct length.
    hash_bits = hasher.hash_length * SIZES[hasher.dtype]

    words_base64 = math.ceil(
        hash_bits / 6)  # Base64 uses 8 bits for every 6 bits
    words_base64 += 0 if words_base64 % 4 == 0 else 4 - (
        words_base64 % 4)  # Base64 always uses multiples of four
    assert len(hash2_1) == words_base64

    words_hex = 2 * math.ceil(
        hash_bits / 8)  # Hex uses 16 bits for every 8 bits
    words_hex += 0 if words_hex % 2 == 0 else 1  # Two characters for every one character.
    assert len(
        hasher.vector_to_string(
            hasher.string_to_vector(hash2_1), hash_format='hex')) == words_hex

    # Verify that low quality images yield zero quality
    image = np.zeros((100, 100, 3)).astype('uint8')
    _, quality = hasher.compute_with_quality(image)
    assert quality == 0

    # Verify that high quality images yield high quality
    # scores.
    assert min(
        hasher.compute_with_quality(filepath)[1]
        for filepath in test_images) == 100

    # Verify that medium quality images yield medium quality
    _, quality = hasher.compute_with_quality(LOW_DETAIL_IMAGE)
    assert 0 < quality < 100

    if opencv_hasher:
        test_opencv_hasher(hasher, image1, image2)
