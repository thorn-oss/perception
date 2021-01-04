# pylint: disable=no-member,invalid-name,too-many-locals,too-many-arguments,too-many-return-statements
import typing
import logging
import concurrent.futures

import numpy as np
import pandas as pd
import tqdm
import cv2

import perception.hashers.tools as pht
import perception.experimental.approximate_deduplication as ad

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_FEATURES = 256
DEFAULT_THRESHOLD = 100
DEFAULT_OVERLAP = 0.01
DEFAULT_MATCH_PCT = 0.4
DEFAULT_INTERSECTION = 0.6
DEFAULT_INLIERS = 5
DEFAULT_MAX_SIZE = 256
DEFAULT_MIN_FEATURES = 10
DEFAULT_RATIO = 0.5


def load_and_preprocess(filepath, max_size=DEFAULT_MAX_SIZE):
    """Read, unletterbox, and resize an image.

    Args:
        filepath: The path to the file
        max_size: The maximum size for a dimension of the image
    """
    image = pht.read(filepath)
    if image is None:
        LOGGER.warning("Failed to load image %s", filepath)
        return None
    res = pht.unletterbox(image)
    if res is None:
        return None
    (x1, x2), (y1, y2) = res
    image = image[y1:y2, x1:x2]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    max_dimension = max(image.shape[:2])
    if max_dimension > max_size:
        scale = max_size / max_dimension
        image = cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    return image


def generate_image_descriptors(
        filepath: str,
        max_features=DEFAULT_MAX_FEATURES,
        min_features=DEFAULT_MIN_FEATURES,
        max_size=DEFAULT_MAX_SIZE
) -> typing.Optional[typing.Tuple[np.array, np.array, typing.Tuple[int, int]]]:
    """Generate SIFT descriptors for a file.

    Args:
        filepath: Path to image file.
        max_features: The maximum number of features to
            extract.
        min_features: The minimum number of features to
            extract.
        max_size: The maximum side length for an image.

    Returns:
        If successful, returns a tuple of keypoints, descriptors,
        and a (width, height) tuple.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    try:
        image = load_and_preprocess(filepath, max_size=max_size)
        if image is None:
            return None
        keypoints, descriptors = sift.detectAndCompute(image, None)
    except FileNotFoundError:
        LOGGER.warning("Image file %s not found.", filepath)
        return None
    except ValueError as e:
        LOGGER.error("Processing image file %s failed.", filepath, exc_info=e)
        return None

    if descriptors is None:
        return None
    if descriptors.shape[0] < min_features:
        return None
    keypoints = np.float32([kp.pt for kp in keypoints])
    return keypoints, descriptors, (image.shape[1], image.shape[0])


def build_reference_df(filepaths: typing.Iterable[str],
                       max_features=DEFAULT_MAX_FEATURES,
                       min_features=DEFAULT_MIN_FEATURES,
                       max_size=DEFAULT_MAX_SIZE) -> pd.DataFrame:
    """Build SIFT descriptors for a list of files.

    Args:
        filepaths: A list of filepaths for which descriptors
            are desired.
        max_features: The maximum number of features to
            extract.
        min_features: The minimum number of features to
            extract.
        max_size: The maximum side length for an image.

    Returns:
        A dataframe, indexed by filepath with columns for descriptors
        and descriptor counts.
    """
    LOGGER.debug("Generating descriptors")
    features = [
        generate_image_descriptors(
            filepath,
            max_features=max_features,
            min_features=min_features,
            max_size=max_size) for filepath in filepaths
    ]
    LOGGER.debug("Finished computing descriptors.")
    return pd.DataFrame({
        'descriptors': [f[1] if f is not None else None for f in features],
        'keypoints': [f[0] if f is not None else None for f in features],
        'descriptor_count':
        [f[1].shape[0] if f is not None else None for f in features],
        'dimensions': [f[2] if f is not None else None for f in features],
        'filepath':
        filepaths
    }).set_index('filepath')


def compute_pairs(reference_df,
                  threshold=DEFAULT_THRESHOLD,
                  minimum_overlap=DEFAULT_OVERLAP,
                  pct_probe=0.1):
    """Compute pairs of matching images from a reference
    dataframe.

    Args:
        reference_df: A dataframe, as computed by build_reference_df.
        threshold: The match threshold between two vectors.
        minimum_overlap: The minimum overlap between a pair of files.
        pct_probe: The percentage of the dataset to search for approximate
            search.
    """
    reference_df = reference_df.dropna(subset=['descriptors'])
    counts = reference_df['descriptor_count'].values.astype('uint32')
    descriptors = np.vstack(reference_df['descriptors'].values)
    pairs = ad.compute_euclidean_pairwise_duplicates_approx(
        X=descriptors.astype('float32'),
        counts=counts,
        threshold=threshold,
        pct_probe=pct_probe,
        minimum_overlap=minimum_overlap)
    return [(reference_df.iloc[p1].name, reference_df.iloc[p2].name)
            for p1, p2 in pairs]


def compute_area(box):
    """Compute the area of a box given a set
    of points x1, y1, x2, y2.

    Args:
        box: A list of coordinates.
    """
    return (box[3] - box[1]) * (box[2] - box[0])


def compute_intersection(kps, filter_arr):
    """Compute the percentage of area covered by
    a set of filtered keypoints versus raw keypoints.

    Args:
        kps: A list of points
        filter_arr: A filter array of same length as kps_raw
            indicating whether to keep that keypoint.
    """
    kps_filtered = kps[filter_arr]
    box_after = np.hstack([kps_filtered.min(axis=0), kps_filtered.max(axis=0)])
    box_before = np.hstack([kps.min(axis=0), kps.max(axis=0)])
    area_before = compute_area(box_before)
    area_after = compute_area(box_after)
    return area_after / area_before


def compute_minimum_intersection(kp1, kp2, filter_arr1, filter_arr2):
    """Compute the minimum intersection between two pairs
    of keypoints (filtered and unfiltered).

    Args:
        kp1: A list of the first set of keypoints
        kp2: A list of the second set of keypoints
        filter_arr1: A filter array for the first set of keypoints
        filter_arr2: A filter array for the second set of keypoints
    """
    return min(
        compute_intersection(kp1, filter_arr1),
        compute_intersection(kp2, filter_arr2))


def validate_match(kp1: np.ndarray,
                   des1: np.ndarray,
                   kp2: np.ndarray,
                   des2: np.ndarray,
                   dims1: typing.Tuple[int, int],
                   dims2: typing.Tuple[int, int],
                   minimum_match: float = DEFAULT_MATCH_PCT,
                   minimum_intersection: float = DEFAULT_INTERSECTION,
                   minimum_inliers: int = DEFAULT_INLIERS,
                   ratio=DEFAULT_RATIO) -> float:
    """Validate the match between two sets of keypoints and descriptors. The
    validation algorithm is as follows:

    #. Compute the mutual set of matches between the two sets of descriptors
       and filter them using Lowe's ratio test.
    #. If the minimum number of passing matches is less than "minimum_match",
       the match fails. This ensures we don't have trivial matches.
    #. Compute the intersection area of the matched keypoints versus the
       raw keypoints. If the area overlap is less than minimum_intersection,
       the match fails. This ensures we don't match on small subsegments of
       an image, such as logos.
    #. Compute a transformation matrix using cv2.findHomography. If we cannot
       obtain a transformation matrix, the match fails. If the sum total
       of inliers for the transformation matrix is less than minimum_inliers,
       the match fails.
    #. Finally, use the transformation matrix on a set of points representing
       the bounding box of each image. If less than minimum_intersection of
       the bounding box fits within the bounds of the transformed version,
       the match fails. This is a second pass safety check for logos and other
       subsegments of images.

    Args:
        kp1: The first set of keypoints
        des1: The first set of descriptors
        kp2: The second set of keypoints
        des2: The second set of descriptors
        dims1: The dimensions (width, height) for the first image
        dims2: The dimensions (width, height) for the second image
        minimum_match: The minimum number of matches passing the ratio test.
        minimum_intersection: The minimum overlapping area between the keypoints
            in the filtered set of matches and the original keypoints.
        minimum_inliers: The minimum number of inliers for the transformation
            matrix.
        ratio: The ratio to use for Lowe's ratio test.

    Returns:
        True if the match passes, False otherwise.
    """
    swap = kp1.shape[0] < kp2.shape[0]
    kpA = kp2 if swap else kp1
    kpB = kp1 if swap else kp2
    dimsA = dims2 if swap else dims1
    dimsB = dims1 if swap else dims2
    desA = des2 if swap else des1
    desB = des1 if swap else des2

    indexA = ad.build_index(desA, approximate=False)
    indexB = ad.build_index(desB, approximate=False)
    if desA is None or indexA is None or desB is None or indexB is None:
        return False
    # pylint: disable=no-value-for-parameter
    distances_A2B, indexes_A2B = indexB.search(desA.astype('float32'), 2)
    distances_B2A, _ = indexA.search(desB.astype('float32'), 2)
    good_A2B, good_B2A = map(
        lambda distances: (distances[:, 0] < distances[:, 1] * ratio),
        [distances_A2B, distances_B2A])
    match = min(good_A2B.sum() / good_A2B.shape[0],
                good_B2A.sum() / good_B2A.shape[0])
    if match < minimum_match:
        # We didn't get enough good matches.
        return False
    kpAM = kpA[good_A2B]
    kpBM = kpB[indexes_A2B[good_A2B, 0]]
    intersection = compute_minimum_intersection(
        kp1=kpA,
        kp2=kpB,
        filter_arr1=good_A2B,
        filter_arr2=indexes_A2B[good_A2B, 0])
    if intersection < minimum_intersection:
        return False
    MAB, mask = cv2.findHomography(
        kpAM.reshape(-1, 1, 2),
        kpBM.reshape(-1, 1, 2),
        cv2.RANSAC,
        1.0,
        maxIters=50_000,
        confidence=0.9999)
    if MAB is None:
        # We didn't get a transformation matrix.
        return False
    if mask.sum() < minimum_inliers:
        # The transformation matrix didn't include enough inliers.
        return False
    # Check how much of each original bounding box fits onto
    # the other image.
    try:
        MBA = np.linalg.inv(MAB)
    except np.linalg.LinAlgError:
        # We couldn't compute the matrix inverse.
        return False
    ptsA = np.array([[0, 0], dimsA]).astype('float32')
    ptsB = np.array([[0, 0], dimsB]).astype('float32')
    ptsAt = cv2.perspectiveTransform(ptsA.reshape((-1, 1, 2)), MAB).reshape(
        -1, 2).clip(0, dimsB)
    ptsBt = cv2.perspectiveTransform(ptsB.reshape((-1, 1, 2)), MBA).reshape(
        -1, 2).clip(0, dimsA)
    bounds_intersection = min(
        np.prod(ptsBt[1] - ptsBt[0]) / np.prod(dimsA),
        np.prod(ptsAt[1] - ptsAt[0]) / np.prod(dimsB),
    )
    if bounds_intersection < minimum_intersection:
        return False
    return True


def deduplicate(
        filepaths: typing.Iterable[str],
        max_features: int = DEFAULT_MAX_FEATURES,
        min_features: int = DEFAULT_MIN_FEATURES,
        max_size: int = DEFAULT_MAX_SIZE,
        coarse_pct_probe: float = ad.DEFAULT_PCT_PROBE,
        coarse_threshold: int = DEFAULT_THRESHOLD,
        minimum_coarse_overlap: float = DEFAULT_OVERLAP,
        minimum_validation_match: float = DEFAULT_MATCH_PCT,
        minimum_validation_intersection: float = DEFAULT_INTERSECTION,
        minimum_validation_inliers: int = DEFAULT_INLIERS,
        ratio: float = DEFAULT_RATIO,
        max_workers: int = None) -> typing.List[typing.Tuple[str, str]]:
    """Deduplicate images by doing the following:

    #. Unletterbox all images and resize to some maximum size, preserving
       aspect ratio.
    #. Compute the SIFT descriptors and keypoints for all the resulting images.
    #. Perform a coarse, approximate search for images with common features.
    #. For each candidate pair, validate it pairwise by checking the features
       and keypoints with the traditional approach using the ratio test. See
       validate_match for more information.

    Args:
        filepaths: The list of images to deduplicate.
        max_features: The maximum number of features to
            extract.
        min_features: The minimum number of features to
            extract.
        max_size: The maximum side length for an image.
        coarse_pct_probe: The minimum fraction of nearest lists to search. If
            the product of pct_probe and the number of lists is less
            than 1, one list will be searched.
        corase_threshold: The threshold for a match as a euclidean distance.
        minimum_coarse_overlap: The minimum overlap between two files to qualify as a match.
        minimum_validation_match: The minimum number of matches passing the ratio test.
        minimum_validation_intersection: The minimum overlapping area between the keypoints
            in the filtered set of matches and the original keypoints.
        minimum_validation_inliers: The minimum number of inliers for the transformation
            matrix.
        ratio: The ratio to use for Lowe's ratio test.
        max_workers: The maximum number of threads to use for doing the final validation
            step.

    Returns:
        A list of pairs of file duplicates.
    """
    reference_df = build_reference_df(
        filepaths=filepaths,
        max_features=max_features,
        min_features=min_features,
        max_size=max_size)
    candidates = compute_pairs(
        reference_df,
        pct_probe=coarse_pct_probe,
        threshold=coarse_threshold,
        minimum_overlap=minimum_coarse_overlap)
    keep = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        batch_size = 10_000
        for start in tqdm.tqdm(range(0, len(candidates), batch_size)):
            futures = {
                executor.submit(
                    validate_match,
                    des1=reference_df.loc[c1]['descriptors'],
                    kp1=reference_df.loc[c1]['keypoints'],
                    des2=reference_df.loc[c2]['descriptors'],
                    kp2=reference_df.loc[c2]['keypoints'],
                    dims1=reference_df.loc[c1]['dimensions'],
                    dims2=reference_df.loc[c2]['dimensions'],
                    minimum_match=minimum_validation_match,
                    minimum_inliers=minimum_validation_inliers,
                    minimum_intersection=minimum_validation_intersection,
                    ratio=ratio): (c1, c2)
                for c1, c2 in candidates[start:start + batch_size]
            }
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    keep.append(futures[future])
    return keep
