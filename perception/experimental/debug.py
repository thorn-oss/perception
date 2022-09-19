from typing import Optional
import logging
import numpy as np
import cv2

import perception.experimental.local_descriptor_deduplication as ldd

LOGGER = logging.getLogger(__name__)

# Set a fixed size for drawing, we don't have the real descriptor size.
KEYPOINT_SIZE: int = 10

# pylint: disable=too-many-locals
def vizualize_pair(
    match_1,
    match_2,
    ratio: float,
    local_path_col: Optional[str] = None,
    sanitized: bool = False,
):
    """Given two rows from a reference df vizualize their overlap.

    Currently recalcs overlap using cv2 default logic.

    Args:
        match_1: The row from a reference df for one image.
        match_2: The row from a reference df for the other image.
        ratio: Value for ratio test, suggest re-using value from matching.
        local_path_col: column in df with path to the image. If None will
            use the index: match_1.name and match_2.name
        sanitized: if True images themselves will not be rendered, only the points.
    Returns:
        An image of the two images concatted together and matching keypoints drawn.
    """
    if local_path_col is not None:
        match_1_path = match_1[local_path_col]
        match_2_path = match_2[local_path_col]
    else:
        match_1_path = match_1.name
        match_2_path = match_2.name

    img1 = np.zeros((match_1.dimensions[1], match_1.dimensions[0], 1), dtype="uint8")
    img2 = np.zeros((match_2.dimensions[1], match_2.dimensions[0], 1), dtype="uint8")

    if not sanitized:
        try:
            img1 = ldd.load_and_preprocess(
                match_1_path, max_size=max(match_1.dimensions), grayscale=False
            )
        except:  # pylint:disable=bare-except
            LOGGER.warning("Failed to load image %s", match_1_path)
        try:
            img2 = ldd.load_and_preprocess(
                match_2_path, max_size=max(match_2.dimensions), grayscale=False
            )
        except:  # pylint:disable=bare-except
            LOGGER.warning("Failed to load image %s", match_2_path)

    # Convert numpy keypoints to cv2.KeyPoints
    kp1_fixed = []
    for k in match_1["keypoints"]:
        kp1_fixed.append(cv2.KeyPoint(k[0], k[1], KEYPOINT_SIZE))

    kp2_fixed = []
    for k in match_2["keypoints"]:
        kp2_fixed.append(cv2.KeyPoint(k[0], k[1], KEYPOINT_SIZE))

    # Note: Could refactor the below to use `good_A2B, good_B2A` from `validate_match_verbose`
    # to ensure stronger overlap in behaviour, but requires
    # changing `validate_match_verbose` to save more details..
    brute_force_matcher = cv2.BFMatcher()
    kn_matches = brute_force_matcher.knnMatch(
        match_1["descriptors"], match_2["descriptors"], k=2
    )
    # Apply ratio test
    good = []
    for nearest_match, next_nearest_match in kn_matches:
        if nearest_match.distance < ratio * next_nearest_match.distance:
            good.append([nearest_match])

    img_matched = cv2.drawMatchesKnn(
        img1,
        kp1_fixed,
        img2,
        kp2_fixed,
        good,
        None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )

    return img_matched
