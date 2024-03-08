import os
import string

import pytest

from perception import hashers, testing
from perception.hashers.image.pdq import PDQHash

TEST_IMAGES = [os.path.join("tests", "images", f"image{n}.jpg") for n in range(1, 11)]


# The PDQ hash isometric computation is inexact. See
# https://github.com/faustomorales/pdqhash-python/blob/master/tests/test_compute.py
# for details.
@pytest.mark.parametrize(
    "hasher_class,pil_opencv_threshold,transform_threshold,opencv_hasher",
    [
        (hashers.AverageHash, 0.1, 0.1, False),
        (hashers.WaveletHash, 0.1, 0.1, False),
        (hashers.PHash, 0.1, 0.1, False),
        (PDQHash, 0.1, 0.15, False),
        (hashers.DHash, 0.1, 0.1, False),
        (hashers.MarrHildreth, 0.1, 0.1, True),
        (hashers.BlockMean, 0.1, 0.1, True),
        (hashers.ColorMoment, 10, 0.1, True),
    ],
)
def test_image_hashing_common(
    hasher_class, pil_opencv_threshold, transform_threshold, opencv_hasher
):
    testing.test_image_hasher_integrity(
        hasher=hasher_class(),
        pil_opencv_threshold=pil_opencv_threshold,
        transform_threshold=transform_threshold,
        opencv_hasher=opencv_hasher,
    )


def test_video_hashing_common():
    testing.test_video_hasher_integrity(
        hasher=hashers.FramewiseHasher(
            frame_hasher=hashers.PHash(hash_size=16),
            interframe_threshold=0.1,
            frames_per_second=1,
        )
    )


def test_video_reading():
    # We should get one red, one green, and one blue frame
    for frame, _, timestamp in hashers.tools.read_video(
        filepath="perception/testing/videos/rgb.m4v", frames_per_second=0.5
    ):
        assert timestamp in [0.0, 2.0, 4.0]
        channel = int(timestamp / 2)
        assert frame[:, :, channel].min() > 230
        for other in [0, 1, 2]:
            if other == channel:
                continue
            assert frame[:, :, other].max() < 20


def test_common_framerate():
    assert hashers.tools.get_common_framerates(
        dict(zip(["a", "b", "c"], [1 / 3, 1 / 2, 1 / 5]))
    ) == {1.0: ("a", "b", "c")}
    assert hashers.tools.get_common_framerates(
        dict(zip(["a", "b", "c"], [1 / 3, 1 / 6, 1 / 9]))
    ) == {1 / 3: ("a", "b", "c")}
    assert hashers.tools.get_common_framerates(
        dict(zip(["a", "b", "c", "d", "e"], [1 / 3, 1 / 2, 1 / 5, 1 / 7, 1 / 11]))
    ) == {1.0: ("a", "b", "c", "d", "e")}
    assert hashers.tools.get_common_framerates(
        dict(zip(string.ascii_lowercase[:6], [10, 5, 3, 1 / 3, 1 / 6, 1 / 9]))
    ) == {3.0: ("c", "d", "e", "f"), 10.0: ("a", "b")}
    assert hashers.tools.get_common_framerates(dict(zip(["a", "b"], [100, 1]))) == {
        100: ("a", "b")
    }


def test_synchronized_hashing():
    video_hashers = {
        "phashframewise": hashers.FramewiseHasher(
            frame_hasher=hashers.PHash(hash_size=16),
            frames_per_second=1,
            interframe_threshold=0.2,
        ),
        "tmkl2": hashers.TMKL2(frames_per_second=15),
        "tmkl1": hashers.TMKL1(frames_per_second=15),
    }

    for filepath in [
        "perception/testing/videos/v1.m4v",
        "perception/testing/videos/v2.m4v",
    ]:
        # Ensure synchronized hashing
        hashes1 = {
            hasher_name: hasher.compute(filepath)
            for hasher_name, hasher in video_hashers.items()
        }
        hashes2 = hashers.tools.compute_synchronized_video_hashes(
            filepath=filepath, hashers=video_hashers
        )
        assert hashes1 == hashes2


def test_scene_detection_batches():
    hasher = hashers.SimpleSceneDetection(
        base_hasher=hashers.TMKL1(
            frames_per_second=30,
            frame_hasher=hashers.PHashU8(),
            norm=None,
            distance_metric="euclidean",
        ),
        max_scene_length=10,
    )
    hashes_v2s = hasher.compute("perception/testing/videos/v2s.mov", errors="raise")
    hashes_batches = []
    frame_count = 0
    for batch in hasher.compute_batches(
        "perception/testing/videos/v2s.mov", batch_size=1
    ):
        for hash_string, frames in batch:
            hashes_batches.append(hash_string)
            frame_count += len(frames)
    # Ensure we get the same hashes whether using compute or compute_batches
    assert len(hashes_batches) == len(hashes_v2s)
    assert all(h1 == h2 for h1, h2 in zip(hashes_batches, hashes_v2s))

    expected_frame_count = 0
    for _, _, _ in hashers.tools.read_video(
        "perception/testing/videos/v2s.mov", frames_per_second=30
    ):
        expected_frame_count += 1

    # Ensure all frames were accounted for in scene detection
    assert expected_frame_count == frame_count


def test_hex_b64_conversion():
    b64_string = (
        """
    CFFRABrAaRKCDQigEBIGwAhNBdIISgVZBxQYAgP4fwYNUR0oBgYCPwwIDSqTAmIH
    FRQhCiT/IT9DpHIeIx4cA2hQcBTwISovFkspMxz/MzdnljeCOEs4LnBYNHHBMC4x
    EC8mPxLaLkI/dywmNk1lMXoqJyCLSyg7BxwRSgTmIlI/LwsrP04hTCMtBSxaGAFB
    """.replace(
            "\n", ""
        )
        .replace(" ", "")
        .strip()
    )
    hex_string = (
        """
    085151001ac06912820d08a0101206c0084d05d2084a05590714180203f87f06
    0d511d280606023f0c080d2a930262071514210a24ff213f43a4721e231e1c03
    68507014f0212a2f164b29331cff333767963782384b382e70583471c1302e31
    102f263f12da2e423f772c26364d65317a2a27208b4b283b071c114a04e62252
    3f2f0b2b3f4e214c232d052c5a180141
    """.replace(
            "\n", ""
        )
        .replace(" ", "")
        .strip()
    )
    assert (
        hashers.tools.hex_to_b64(hex_string, dtype="uint8", hash_length=144)
        == b64_string
    )
    assert (
        hashers.tools.b64_to_hex(b64_string, dtype="uint8", hash_length=144)
        == hex_string
    )


class TestSimpleSceneDetection:
    def test_compute_where_base_hasher_returns_multiple(self):
        # Establish the hasher
        hasher = hashers.SimpleSceneDetection(
            base_hasher=hashers.TMKL1(
                frames_per_second=30,
                frame_hasher=hashers.PHashU8(),
                norm=None,
                distance_metric="euclidean",
            ),
            max_scene_length=10,
        )
        # Confirm it's configured as we'd like it to be. By default,
        # base_hasher.returns_multiple is False, but we'll force it to be
        # True in order to ensure the existing logic handles things gracefully
        hasher.base_hasher.returns_multiple = True
        # This assertion is likely unnecessary, but it helps us a) confirm that
        # the underlying assumptions for this test hold true and b) protect against
        # regression, in case something bizarre happens with returns_multiple down the line
        assert hasher.base_hasher.returns_multiple is True

        # Confirm the results of compute look like we'd expect them to. In this case
        # we've shoehorned an invalid config in, so we don't particularly care about
        # the values, so much as the fact that the logic parses the results gracefully
        hashes = hasher.compute("perception/testing/videos/v2s.mov", errors="raise")
        assert hashes

    def test_compute_where_base_hasher_does_not_return_multiple(self):
        # Establish the hasher
        hasher = hashers.SimpleSceneDetection(
            base_hasher=hashers.TMKL1(
                frames_per_second=30,
                frame_hasher=hashers.PHashU8(),
                norm=None,
                distance_metric="euclidean",
            ),
            max_scene_length=10,
        )
        hasher.base_hasher.returns_multiple = False

        # Confirm the hasher is configured as we'd like it to be
        assert hasher.base_hasher.returns_multiple is False

        # Confirm the results of compute look like we'd expect them to
        assert (
            len(hasher.compute("perception/testing/videos/v1.m4v", errors="raise")) == 1
        )
        assert (
            len(hasher.compute("perception/testing/videos/v2.m4v", errors="raise")) == 1
        )
        hashes_v2s = hasher.compute("perception/testing/videos/v2s.mov", errors="raise")
        assert len(hashes_v2s) == 2

    def test_compute_batches_where_base_hasher_returns_multiple(self):
        hasher = hashers.SimpleSceneDetection(
            base_hasher=hashers.TMKL1(
                frames_per_second=30,
                frame_hasher=hashers.PHashU8(),
                norm=None,
                distance_metric="euclidean",
            ),
            max_scene_length=10,
        )
        # Confirm it's configured as we'd like it to be. By default,
        # base_hasher.returns_multiple is False, but we'll force it to be
        # True in order to ensure the existing logic handles things gracefully
        hasher.base_hasher.returns_multiple = True
        # This assertion is likely unnecessary, but it helps us a) confirm that
        # the underlying assumptions for this test hold true and b) protect against
        # regression, in case something bizarre happens with returns_multiple down the line
        assert hasher.base_hasher.returns_multiple is True

        # Confirm the results of compute look like we'd expect them to. In this case
        # we've shoehorned an invalid config in, so we don't particularly care about
        # the values, so much as the fact that the logic parses the results gracefully
        hashes = hasher.compute_batches(
            "perception/testing/videos/v2s.mov", batch_size=1
        )
        assert hashes

    def test_compute_batches_where_base_hasher_does_not_return_multiple(self):
        hasher = hashers.SimpleSceneDetection(
            base_hasher=hashers.TMKL1(
                frames_per_second=30,
                frame_hasher=hashers.PHashU8(),
                norm=None,
                distance_metric="euclidean",
            ),
            max_scene_length=10,
        )
        hasher.base_hasher.returns_multiple = False

        assert hasher.base_hasher.returns_multiple is False
        hashes_v2s = hasher.compute("perception/testing/videos/v2s.mov", errors="raise")
        hashes_batches = []
        frame_count = 0
        for batch in hasher.compute_batches(
            "perception/testing/videos/v2s.mov", batch_size=1
        ):
            for hash_string, frames in batch:
                hashes_batches.append(hash_string)
                frame_count += len(frames)
        # Ensure we get the same hashes whether using compute or compute_batches
        assert len(hashes_batches) == len(hashes_v2s)
        assert all(h1 == h2 for h1, h2 in zip(hashes_batches, hashes_v2s))

        expected_frame_count = 0
        for _, _, _ in hashers.tools.read_video(
            "perception/testing/videos/v2s.mov", frames_per_second=30
        ):
            expected_frame_count += 1

        # Ensure all frames were accounted for in scene detection
        assert expected_frame_count == frame_count

    def test_compute_with_timestamps(self):
        hasher = hashers.SimpleSceneDetection(
            base_hasher=hashers.TMKL1(
                frames_per_second=30,
                frame_hasher=hashers.PHashU8(),
                norm=None,
                distance_metric="euclidean",
            ),
            max_scene_length=10,
        )
        hashes = hasher.compute_with_timestamps(
            "perception/testing/videos/v2s.mov", errors="raise"
        )

        # Confirm we have two hashes
        assert len(hashes) == 2
        scene_1_hash = hashes[0]
        scene_2_hash = hashes[1]

        # Sanity check timestamps within a given scene
        assert scene_1_hash["start_timestamp"] < scene_1_hash["end_timestamp"]
        assert scene_2_hash["start_timestamp"] < scene_2_hash["end_timestamp"]

        # Sanity check timestamps between scenes
        assert scene_1_hash["end_timestamp"] <= scene_2_hash["start_timestamp"]

        # Sanity check frame index between scenes
        assert scene_1_hash["frame_index"] <= scene_2_hash["frame_index"]
