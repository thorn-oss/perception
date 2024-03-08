import os
import shutil
import tempfile

import numpy as np
import pytest

from perception import hashers, testing, tools


def test_deduplicate():
    directory = tempfile.TemporaryDirectory()
    original = testing.DEFAULT_TEST_IMAGES[0]
    duplicate = os.path.join(directory.name, "image1.jpg")
    shutil.copy(original, duplicate)
    pairs = tools.deduplicate(
        files=[
            testing.DEFAULT_TEST_IMAGES[0],
            testing.DEFAULT_TEST_IMAGES[1],
            duplicate,
        ],
        hashers=[(hashers.PHash(hash_size=16), 0.25)],
    )
    assert len(pairs) == 1
    file1, file2 = pairs[0]
    assert ((file1 == duplicate) and (file2 == original)) or (
        (file1 == original) and (file2 == duplicate)
    )


def test_deduplicate_u8():
    # This test verifies that extensions.compute_euclidean_pairwise_duplicates
    # works properly.
    directory = tempfile.TemporaryDirectory()
    original = testing.DEFAULT_TEST_IMAGES[0]
    duplicate = os.path.join(directory.name, "image1.jpg")
    shutil.copy(original, duplicate)
    pairs = tools.deduplicate(
        files=[
            testing.DEFAULT_TEST_IMAGES[0],
            testing.DEFAULT_TEST_IMAGES[1],
            duplicate,
        ],
        hashers=[(hashers.PHashU8(hash_size=16), 10)],
    )
    assert len(pairs) == 1
    file1, file2 = pairs[0]
    assert ((file1 == duplicate) and (file2 == original)) or (
        (file1 == original) and (file2 == duplicate)
    )


def test_deduplicate_hashes_multiple():
    # This test verifies that deduplicate_hashes functions properly
    # when there is more than one hash for a file.
    directory = tempfile.TemporaryDirectory()
    original = testing.DEFAULT_TEST_IMAGES[0]
    duplicate = os.path.join(directory.name, "image1.jpg")
    hasher = hashers.PHashU8(hash_size=16)
    shutil.copy(original, duplicate)
    hashes = [
        (0, hasher.compute(original)),
        (1, hasher.compute(duplicate)),
        (1, hasher.compute(duplicate)),
        (1, hasher.compute(duplicate)),
        (2, hasher.compute(testing.DEFAULT_TEST_IMAGES[1])),
    ]
    pairs = tools.deduplicate_hashes(
        hashes=hashes,
        threshold=10,
        hash_format="base64",
        hash_length=hasher.hash_length,
        distance_metric="euclidean",
        hash_dtype="uint8",
    )
    assert len(pairs) == 1
    file1, file2 = pairs[0]
    assert ((file1 == 0) and (file2 == 1)) or ((file1 == 1) and (file2 == 0))


def test_compute_euclidean_pairwise_duplicates():
    # The purpose of this test is to verify that the handling of
    # deduplication with files that have multiple hashes works
    # properly. This is particularly important for video where
    # we are likely to have many hashes.
    X = np.array(
        [
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
            [6, 6, 6],
        ]
    )

    # Use grouped files.
    counts = np.array([3, 3, 2, 2])
    expected = np.array(
        [[2 / 3, 2 / 3], [0, 0], [0, 0], [1 / 3, 1 / 2], [0, 0], [0, 0]]
    )
    actual = tools.extensions.compute_euclidean_pairwise_duplicates(
        X=X.astype("int32"),
        threshold=1,
        counts=counts.astype("uint32"),
        compute_overlap=True,
    )
    assert (expected == actual).all()

    # Use without computing overlap.
    expected = np.array([[2, 2], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0]])
    actual = tools.extensions.compute_euclidean_pairwise_duplicates(
        X=X.astype("int32"),
        threshold=1,
        counts=counts.astype("uint32"),
        compute_overlap=False,
    )
    assert (expected == actual).all()

    # Use ungrouped files.
    X = np.array(
        [
            # File 1
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [1, 1, 1],
        ]
    )
    expected = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0]])
    actual = tools.extensions.compute_euclidean_pairwise_duplicates(
        X=X.astype("int32"), threshold=1, compute_overlap=True
    )
    assert (expected == actual).all()


def test_api_is_over_https():
    matcher_https = tools.SaferMatcher(api_key="foo", url="https://www.example.com/")
    assert matcher_https

    if "SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP" in os.environ:
        del os.environ["SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP"]
    with pytest.raises(ValueError):
        tools.SaferMatcher(api_key="foo", url="http://www.example.com/")

    os.environ["SAFER_MATCHING_SERVICE_DEV_ALLOW_HTTP"] = "1"
    matcher_http_with_escape_hatch = tools.SaferMatcher(
        api_key="foo", url="http://www.example.com/"
    )
    assert matcher_http_with_escape_hatch


def test_unletterbox():
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    padded = np.zeros((image.shape[0] + 100, image.shape[1] + 50, 3), dtype="uint8")
    padded[50 : 50 + image.shape[0], 25 : 25 + image.shape[1]] = image
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded)
    assert y1 == 50
    assert y2 == 50 + image.shape[0]
    assert x1 == 25
    assert x2 == 25 + image.shape[1]


def test_unletterbox_color():
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    padded = np.zeros((image.shape[0] + 100, image.shape[1] + 50, 3), dtype="uint8")
    padded[:, :] = (200, 0, 200)
    padded[50 : 50 + image.shape[0], 25 : 25 + image.shape[1]] = image
    # Should not unletterbox since not black.
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded, only_remove_black=True)
    assert y1 == 0
    assert y2 == padded.shape[0]
    assert x1 == 0
    assert x2 == padded.shape[1]

    # Should  unletterbox color:
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded, only_remove_black=False)
    assert y1 == 50
    assert y2 == 50 + image.shape[0]
    assert x1 == 25
    assert x2 == 25 + image.shape[1]


def test_unletterbox_aspect_ratio():
    """Test the value of .1 in unletterbox()."""
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    h, w, z = image.shape

    # make tall skinny images with non-trivial content just below and
    # above 10% threshold
    base = int(4.5 * h)  # 2 * base + h = 100%
    h_fail, h_pass = base + 10, base - 10

    padded = np.r_[np.zeros((h_fail, w, 3)), image, np.zeros((h_fail, w, 3))]
    assert None is hashers.tools.unletterbox(padded)

    padded = np.r_[np.zeros((h_pass, w, 3)), image, np.zeros((h_pass, w, 3))]
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(padded)

    assert y1 == h_pass
    assert y2 == h_pass + image.shape[0]
    assert x1 == 0
    assert x2 == image.shape[1]


def test_unletterbox_noblackbars():
    image = hashers.tools.read(testing.DEFAULT_TEST_IMAGES[0])
    (x1, x2), (y1, y2) = hashers.tools.unletterbox(image)
    assert x1 == 0
    assert y1 == 0
    assert x2 == image.shape[1]
    assert y2 == image.shape[0]


def test_ffmpeg_video():
    """Check that the FFMPEG video parsing code provides substantially similar
    results to the OpenCV approach (which also uses FFMPEG under the hood but
    also has different frame selection logic)."""
    frames_per_second = 2.3
    for filepath in testing.DEFAULT_TEST_VIDEOS:
        filename = os.path.basename(filepath)
        for (frame1, index1, timestamp1), (frame2, index2, timestamp2) in zip(
            hashers.tools.read_video_to_generator_ffmpeg(
                filepath, frames_per_second=frames_per_second
            ),
            hashers.tools.read_video_to_generator(
                filepath, frames_per_second=frames_per_second
            ),
        ):
            diff = np.abs(frame1.astype("int32") - frame2.astype("int32")).flatten()
            assert index1 == index2, f"Index mismatch for {filename}"
            np.testing.assert_allclose(
                timestamp1, timestamp2
            ), f"Timestamp mismatch for {filename}"
            assert np.percentile(diff, 75) < 25, f"Frame mismatch for {filename}"
