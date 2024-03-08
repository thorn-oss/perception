import logging
import random
from typing import Optional

import cv2
import numpy as np

import perception.experimental.local_descriptor_deduplication as ldd

LOGGER = logging.getLogger(__name__)

# Set a fixed size for drawing, we don't have the real descriptor size.
KEYPOINT_SIZE: int = 8


def vizualize_pair(
    features_1,
    features_2,
    ratio: float,
    match_metadata=None,
    local_path_col: Optional[str] = None,
    sanitized: bool = False,
):
    """Given two rows from a reference df vizualize their overlap.

    Currently recalcs overlap using cv2 default logic.

    Args:
        features_1: The row from a reference df for one image.
        features_2: The row from a reference df for the other image.
        ratio: Value for ratio test, suggest re-using value from matching.
        match_metadata: metadata returned from matching, if None will redo brute force matching.
        local_path_col: column in df with path to the image. If None will
            use the index: features_1.name and features_2.name
        sanitized: if True images themselves will not be rendered, only the points.
    Returns:
        An image of the two images concatted together and matching keypoints drawn.
    """
    # Set a fixed size for drawing, we don't have the real descriptor size.
    if local_path_col is not None:
        features_1_path = features_1[local_path_col]
        features_2_path = features_2[local_path_col]
    else:
        features_1_path = features_1.name
        features_2_path = features_2.name

    img1 = np.zeros(
        (features_1.dimensions[1], features_1.dimensions[0], 1), dtype="uint8"
    )
    img2 = np.zeros(
        (features_2.dimensions[1], features_2.dimensions[0], 1), dtype="uint8"
    )

    if not sanitized:
        try:
            img1 = ldd.load_and_preprocess(
                features_1_path, max_size=max(features_1.dimensions), grayscale=False
            )
        except Exception:
            LOGGER.warning("Failed to load image %s", features_1_path)
        try:
            img2 = ldd.load_and_preprocess(
                features_2_path, max_size=max(features_2.dimensions), grayscale=False
            )
        except Exception:
            LOGGER.warning("Failed to load image %s", features_2_path)

    if match_metadata is not None:
        img_matched = viz_match_data(features_1, features_2, img1, img2, match_metadata)
    else:
        LOGGER.warning(
            """No match_metadata provided, recalculating match points,
            won't match perception match points."""
        )
        img_matched = viz_brute_force(features_1, features_2, img1, img2, ratio=ratio)

    return img_matched


def viz_match_data(features_1, features_2, img1, img2, match_metadata):
    """Given match data viz matching points.

    Args:
        features_1: The row from a reference df for one image.
        features_2: The row from a reference df for the other image.
        img1: cv2 of first image
        img2: cv2 of second image
        match_metadata: metadata returned from matching, if None will redo
            brute force matching.
    Returns:
        cv2 img with matching keypoints drawn.
    """
    # NOTE: could refactor to put matches in to correct format and use: cv2.drawMatchesKnn,
    #  but python docs on necessary class not clear.

    # Pad img1 or img2 vertically with black pixels to match the height of the other image
    if img1.shape[0] > img2.shape[0]:
        img2 = np.pad(
            img2,
            ((0, img1.shape[0] - img2.shape[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif img1.shape[0] < img2.shape[0]:
        img1 = np.pad(
            img1,
            ((0, img2.shape[0] - img1.shape[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    # draw two images h concat:
    img_matched = np.concatenate((img1, img2), axis=1)
    # draw all points in kp_1
    for k in features_1["keypoints"]:
        new_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        cv2.circle(img_matched, (int(k[0]), int(k[1])), KEYPOINT_SIZE, new_color, 1)
    # draw all points in kp_2
    for k in features_2["keypoints"]:
        new_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        cv2.circle(
            img_matched,
            (int(k[0] + features_1.dimensions[0]), int(k[1])),
            KEYPOINT_SIZE,
            new_color,
            1,
        )

    # draw lines between matching points
    for i in range(len(match_metadata["final_matched_b_pts"])):
        new_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        a_pt = (
            int(match_metadata["final_matched_a_pts"][i][0]),
            int(match_metadata["final_matched_a_pts"][i][1]),
        )
        b_pt = (
            int(match_metadata["final_matched_b_pts"][i][0] + features_1.dimensions[0]),
            int(match_metadata["final_matched_b_pts"][i][1]),
        )
        cv2.circle(img_matched, a_pt, KEYPOINT_SIZE, new_color, 2)
        cv2.circle(img_matched, b_pt, KEYPOINT_SIZE, new_color, 2)
        cv2.line(
            img_matched,
            a_pt,
            b_pt,
            new_color,
            1,
        )
    return img_matched


def viz_brute_force(features_1, features_2, img1, img2, ratio: float):
    """
    Given two rows from a reference df vizualize their overlap.

    NOTE: It redoes matching using cv2 bruteforce, so will not match the same
        as the perception matching code.

    Args:
        features_1: The row from a reference df for one image.
        features_2: The row from a reference df for the other image.
        img1: cv2 of first image
        img2: cv2 of second image
        ratio: Value for ratio test, suggest re-using value from matching.

    Returns:
        An image of the two images concatted together and matching keypoints drawn.
    """
    # Convert numpy keypoints to cv2.KeyPoints
    kp1_fixed = []
    for k in features_1["keypoints"]:
        kp1_fixed.append(cv2.KeyPoint(k[0], k[1], KEYPOINT_SIZE))

    kp2_fixed = []
    for k in features_2["keypoints"]:
        kp2_fixed.append(cv2.KeyPoint(k[0], k[1], KEYPOINT_SIZE))
    brute_force_matcher = cv2.BFMatcher()
    kn_matches = brute_force_matcher.knnMatch(
        features_1["descriptors"], features_2["descriptors"], k=2
    )
    # Apply ratio test
    good = []
    for nearest_match, next_nearest_match in kn_matches:
        if nearest_match.distance < ratio * next_nearest_match.distance:
            good.append([nearest_match])
    img_matched = cv2.drawMatchesKnn(  # type: ignore[call-overload]
        img1,
        kp1_fixed,
        img2,
        kp2_fixed,
        good,
        None,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
    return img_matched
