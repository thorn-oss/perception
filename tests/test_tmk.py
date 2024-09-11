import gzip
import json
from pathlib import Path
from typing import cast
import platform

import numpy as np
import pytest

from perception.hashers.video import tmk

TEST_FILES = Path("perception") / "testing" / "videos"


def test_tmk_parity():
    if platform.machine() == "arm64":
        pytest.xfail("TMK is not supported on ARM64")

    hasher = tmk.TMKL2()
    with gzip.open(TEST_FILES / "expected_tmk.json.gz", "rt", encoding="utf8") as f:
        expected_output = json.load(f)
    expected_output = {k: np.array(v) for k, v in expected_output.items()}

    output = []

    for filepath in [
        "perception/testing/videos/v1.m4v",
        "perception/testing/videos/v2.m4v",
    ]:
        hash_value: np.ndarray = cast(
            np.ndarray, hasher.compute(filepath=filepath, hash_format="vector")
        )
        output.append(hash_value.reshape((4, 64, -1)))

    # Verify the hashes are the same
    for o, t in zip(output, expected_output["hashes"]):
        np.testing.assert_allclose(o.reshape(*t.shape), t)

    # Verify the pair-wise scores are the same
    offsets = np.arange(-5, 5)
    for normalization in ["feat", "feat_freq", "matrix"]:
        score = hasher._score_pair(
            output[0], output[1], offsets=offsets, normalization=normalization
        )
        np.testing.assert_allclose(score, expected_output[normalization])
