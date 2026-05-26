import gzip
import json
from pathlib import Path
from typing import cast

import numpy as np

from perception.hashers.video import tmk

TEST_FILES = Path("perception") / "testing" / "videos"


def test_tmk_parity():
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

    # Verify the hashes have the correct shape and similar magnitude
    for o, t in zip(output, expected_output["hashes"]):
        assert o.shape == t.shape
        # Cosine similarity between flattened hashes should be very high
        cos_sim = np.dot(o.flatten(), t.flatten()) / (
            np.linalg.norm(o) * np.linalg.norm(t)
        )
        assert cos_sim > 0.99, f"Hash cosine similarity too low: {cos_sim}"

    # Verify the pair-wise scores are the same (this is what matters for matching)
    offsets = np.arange(-5, 5)
    for normalization in ["feat", "feat_freq", "matrix"]:
        score = hasher._score_pair(
            output[0], output[1], offsets=offsets, normalization=normalization
        )
        np.testing.assert_allclose(score, expected_output[normalization], atol=0.05)
