# pylint: disable=invalid-name,too-many-locals
import os
import typing
import pkg_resources

import cv2
import pytest
import numpy as np
import pandas as pd
from PIL import Image  # pylint: disable=import-error

from .. import hashers

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
    pkg_resources.resource_filename('perception',
                                    os.path.join('images', f'image{n}.jpg'))
    for n in range(1, 11)
]


@typing.no_type_check
def test_opencv_hasher(hasher: hashers.Hasher, image1: str, image2: str):
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


def test_hasher_integrity(hasher: hashers.Hasher,
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
    images_1x = test_images
    images_10x = images_1x * 10
    if not hasher.allow_parallel:
        with pytest.warns(UserWarning, match='cannot be used in parallel'):
            hashes_parallel = pd.DataFrame.from_records(
                hasher.compute_parallel(images_10x, max_workers=5))
    else:
        hashes_parallel = pd.DataFrame.from_records(
            hasher.compute_parallel(images_10x, max_workers=5))

    hashes_sequential = pd.DataFrame.from_records([{
        'hash':
        hasher.compute(image),
        'filepath':
        image,
        'error':
        None
    } for image in images_10x])
    merged = hashes_sequential.merge(hashes_parallel, on='filepath')
    assert hashes_parallel['error'].isnull().all()
    assert (merged['hash_x'] == merged['hash_y']).all()

    # Ensure the isometric hashes computation work properly
    for image in images_1x:
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
    assert len(hash2_1) == hashers.tools.get_string_length(
        hash_length=hasher.hash_length,
        dtype=hasher.dtype,
        hash_format='base64')
    assert len(
        hasher.vector_to_string(
            hasher.string_to_vector(hash2_1),
            hash_format='hex')) == hashers.tools.get_string_length(
                hash_length=hasher.hash_length,
                dtype=hasher.dtype,
                hash_format='hex')

    # Verify that low quality images yield zero quality
    image = np.zeros((100, 100, 3)).astype('uint8')
    _, quality = hasher.compute_with_quality(image)
    assert quality == 0

    # Verify that high quality images yield high quality
    # scores.
    assert min(
        hasher.compute_with_quality(filepath)[1]
        for filepath in images_1x) == 100

    # Verify that medium quality images yield medium quality
    _, quality = hasher.compute_with_quality(LOW_DETAIL_IMAGE)
    assert 0 < quality < 100

    if opencv_hasher:
        test_opencv_hasher(hasher, image1, image2)
