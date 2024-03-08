import numpy as np
import torch
import videoalignment.models
import videoalignment.test_models

from perception.hashers.tools import read_video
from perception.hashers.video import tmk


def test_tmk_parity():
    hasher = tmk.TMKL2()

    ours = []
    theirs = []

    for filepath in [
        "perception/testing/videos/v1.m4v",
        "perception/testing/videos/v2.m4v",
    ]:
        features_timestamps = [
            (hasher.frame_hasher.compute(frame, hash_format="vector"), timestamp)
            for frame, _, timestamp in read_video(
                filepath=filepath,
                frames_per_second=hasher.frames_per_second,
                errors="raise",
            )
        ]
        features = np.array([features for features, _ in features_timestamps])
        timestamps = np.array([timestamp for _, timestamp in features_timestamps])
        model = videoalignment.models.TMK_Poullot(
            videoalignment.test_models.ModelArgs(m=32)
        )
        theirs.append(
            model.single_fv(
                ts=torch.from_numpy(features[np.newaxis,]),
                xs=torch.from_numpy(timestamps[np.newaxis,]),
            ).numpy()[0]
        )
        ours.append(hasher.compute(filepath=filepath, hash_format="vector"))

    # Verify the hashes are the same
    for o, t in zip(ours, theirs):
        np.testing.assert_allclose(o.reshape(*t.shape), t, rtol=0.05)

    offsets = np.arange(-5, 5)

    # Verify the pair-wise scores are the same
    offsets = np.arange(-5, 5)
    for normalization in ["feat", "feat_freq", "matrix"]:
        model.tmk.normalization = normalization
        scores_theirs = model.tmk.merge(
            fv_a=torch.from_numpy(theirs[0][np.newaxis,]),
            fv_b=torch.from_numpy(theirs[1][np.newaxis,]),
            offsets=torch.from_numpy(offsets[np.newaxis,]),
        )[0]
        scores_ours = hasher._score_pair(
            theirs[0], theirs[1], offsets=offsets, normalization=normalization
        )
        np.testing.assert_allclose(scores_ours, scores_theirs)
