import os
import tempfile

import cv2
import imgaug
import pandas as pd
import pytest

import perception.benchmarking.image as pb
import perception.benchmarking.image_transforms as pbit
import perception.experimental.approximate_deduplication as ad
import perception.experimental.local_descriptor_deduplication as ldd
import perception.hashers.tools as pht
import perception.testing as pt
from perception.experimental.debug import vizualize_pair

# Params for object level matching.
OBJECT_MATCH_PARAMS = {
    "strong_match_threshold": 0.3,  # Ideally something close to 95% precision.
    "ratio": 0.5,
    "coarse_pct_probe": 0.1,
    "minimum_coarse_overlap": 0.001,
    "coarse_threshold": 100.0,
    "minimum_validation_match": 0.04,
    "minimum_validation_intersection": 0.04,
    "minimum_validation_inliers": 6,
}


@pytest.mark.parametrize("hasher", [ldd.SIFT(), ldd.AKAZE()])
def test_deduplication(hasher):
    tdir = tempfile.TemporaryDirectory()
    watermark = cv2.cvtColor(
        cv2.imread(pt.DEFAULT_TEST_LOGOS[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
    )
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, "test") for filepath in pt.DEFAULT_TEST_IMAGES]
    ).transform(
        transforms={
            "noop": lambda image: image,
            "pad": imgaug.augmenters.Pad(percent=0.1),
            "crop": imgaug.augmenters.Crop(percent=0.1),
            "watermark": pbit.apply_watermark(watermark, alpha=1, size=0.8),
        },
        storage_dir=tdir.name,
    )
    df = transformed._df.set_index("filepath")
    pairs = ldd.deduplicate(
        filepaths_or_reference_df=df.index, max_workers=2, hasher=hasher
    )  #  Test throws errors if unset.
    clustered = (
        pd.DataFrame(ad.pairs_to_clusters(ids=df.index, pairs=pairs))
        .set_index("id")
        .merge(df, left_index=True, right_index=True)
        .reset_index()
    )
    n_clusters = clustered["cluster"].nunique()
    n_transforms = clustered["transform_name"].nunique()
    perfect = (
        clustered.groupby("cluster")
        .apply(
            lambda g: g["guid"].nunique() == 1
            and g["transform_name"].nunique() == n_transforms
        )
        .sum()
    )

    tainted = clustered.groupby("cluster")["guid"].nunique().gt(1).sum()
    pct_perfect = perfect / n_clusters
    pct_tainted = tainted / n_clusters
    assert pct_tainted == 0
    assert pct_perfect > 0.1


@pytest.mark.parametrize("hasher", [ldd.SIFT(), ldd.AKAZE()])
def test_deduplication_across_sets(hasher):
    tdir = tempfile.TemporaryDirectory()
    watermark = cv2.cvtColor(
        cv2.imread(pt.DEFAULT_TEST_LOGOS[0], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
    )
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, "test") for filepath in pt.DEFAULT_TEST_IMAGES]
    ).transform(
        transforms={
            "noop": lambda image: image,
            "pad": imgaug.augmenters.Pad(percent=0.1),
            "crop": imgaug.augmenters.Crop(percent=0.1),
            "watermark": pbit.apply_watermark(watermark, alpha=1, size=0.8),
        },
        storage_dir=tdir.name,
    )

    df = transformed._df.set_index("filepath")
    query_images = list(df[df.transform_name == "noop"].index.values)
    images_to_match_to = list(df[~(df.transform_name == "noop")].index.values)

    pairs = ldd.deduplicate(
        filepaths_or_reference_df=images_to_match_to,
        query_filepaths_or_df=query_images,
        max_workers=2,
        hasher=hasher,
    )  #  Test throws errors if unset.

    assert len(pairs) >= 20, "Wrong # of pairs."
    only_one_noop = [p for p in pairs if (("noop" in p[0]) != ("noop" in p[1]))]
    assert len(only_one_noop) == len(
        pairs
    ), "All pairs must be between a noop and non-noop file"


@pytest.mark.parametrize("hasher", [ldd.SIFT(), ldd.AKAZE()])
def test_validation_for_overlapping_case(hasher):
    tdir = tempfile.TemporaryDirectory()
    # Each image will have the center of the other
    # pasted in the top left corner.
    image1 = pht.read(pt.DEFAULT_TEST_IMAGES[0])
    image2 = pht.read(pt.DEFAULT_TEST_IMAGES[1])
    image1[:100, :100] = image2[100:200, 100:200]
    image2[:100, :100] = image1[100:200, 100:200]
    fp1 = os.path.join(tdir.name, "test1.jpg")
    fp2 = os.path.join(tdir.name, "test2.jpg")
    cv2.imwrite(fp1, image1[..., ::-1])
    cv2.imwrite(fp2, image2[..., ::-1])
    descriptor1 = ldd.generate_image_descriptors(fp1, hasher)
    descriptor2 = ldd.generate_image_descriptors(fp2, hasher)
    assert descriptor1 is not None
    assert descriptor2 is not None

    # These images should not match.
    assert not hasher.validate_match(descriptor1=descriptor1, descriptor2=descriptor2)[
        0
    ]


@pytest.mark.parametrize("hasher", [ldd.SIFT(), ldd.AKAZE()])
def test_handling_bad_file_case(caplog, hasher):
    tdir = tempfile.TemporaryDirectory()
    missing_file = os.path.join(tdir.name, "missing-file")
    bad_file_handle = tempfile.NamedTemporaryFile()
    bad_file = bad_file_handle.name
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, "test") for filepath in pt.DEFAULT_TEST_IMAGES]
    ).transform(
        transforms={
            "noop": lambda image: image,
        },
        storage_dir=tdir.name,
    )
    df = transformed._df.set_index("filepath")
    df.loc[missing_file] = df.iloc[0]
    df.loc[bad_file] = df.iloc[0]
    pairs = ldd.deduplicate(filepaths_or_reference_df=df.index, hasher=hasher)
    clustered = (
        pd.DataFrame(ad.pairs_to_clusters(ids=df.index, pairs=pairs))
        .set_index("id")
        .merge(df, left_index=True, right_index=True)
        .reset_index()
    )

    assert bad_file not in clustered.index
    assert missing_file not in clustered.index

    bad_file_error = next(
        record for record in caplog.records if bad_file in record.message
    )
    assert bad_file_error
    assert bad_file_error.levelname == "ERROR"

    missing_file_warning = next(
        record for record in caplog.records if missing_file in record.message
    )
    assert missing_file_warning
    assert missing_file_warning.levelname == "WARNING"


def test_handling_hasher_mismatch():
    tdir = tempfile.TemporaryDirectory()
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, "test") for filepath in pt.DEFAULT_TEST_IMAGES]
    ).transform(
        transforms={
            "noop": lambda image: image,
        },
        storage_dir=tdir.name,
    )
    df = transformed._df.set_index("filepath")
    reference_df = ldd.build_reference_df(filepaths=df.index, hasher=ldd.SIFT())
    query_df = ldd.build_reference_df(filepaths=df.index, hasher=ldd.AKAZE())
    with pytest.raises(AssertionError):
        ldd.deduplicate(reference_df, query_df)


def test_viz_pair():
    object_sift = ldd.SIFT(
        max_features=256,
        ratio=OBJECT_MATCH_PARAMS["ratio"],
        threshold=OBJECT_MATCH_PARAMS["coarse_threshold"],
        overlap=OBJECT_MATCH_PARAMS["minimum_coarse_overlap"],
        validation_match=OBJECT_MATCH_PARAMS["minimum_validation_match"],
        validation_inliers=OBJECT_MATCH_PARAMS["minimum_validation_inliers"],
        validation_intersection=OBJECT_MATCH_PARAMS["minimum_validation_intersection"],
    )
    filepaths = [
        "tests/images/chair.png",
        "tests/images/chair3.png",
        "tests/images/chair-square.png",
        "tests/images/chair-tall.png",
    ]
    reference_df = ldd.build_reference_df(
        filepaths=filepaths,
        hasher=object_sift,
        min_features=10,
        max_size=1000,
        show_progress=False,
    )
    pairs = ldd.deduplicate(
        filepaths_or_reference_df=reference_df,
        hasher=object_sift,
        max_size=1000,
        min_features=10,
        verbose=True,
    )
    row = pairs[0]
    viz_img = vizualize_pair(
        reference_df.loc[row[0]],
        reference_df.loc[row[1]],
        0.5,
        match_metadata=row[2],
        sanitized=False,
    )
    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/images/debug-image.png", viz_img)


def test_viz_pair_symmetry():
    # This test catches a regression where if the smaller image was the query one LDD would swap
    # points during distance calculation, but not unswap points before returning them.
    object_sift = ldd.SIFT(
        max_features=256,
        ratio=OBJECT_MATCH_PARAMS["ratio"],
        threshold=OBJECT_MATCH_PARAMS["coarse_threshold"],
        overlap=OBJECT_MATCH_PARAMS["minimum_coarse_overlap"],
        validation_match=OBJECT_MATCH_PARAMS["minimum_validation_match"],
        validation_inliers=OBJECT_MATCH_PARAMS["minimum_validation_inliers"],
        validation_intersection=OBJECT_MATCH_PARAMS["minimum_validation_intersection"],
    )
    filepaths = [
        "tests/images/chair.png",
        "tests/images/chair3.png",
    ]
    reference_df = ldd.build_reference_df(
        filepaths=filepaths,
        hasher=object_sift,
        min_features=10,
        max_size=1000,
        show_progress=False,
    )
    pairs = ldd.deduplicate(
        filepaths_or_reference_df=filepaths[:1],
        query_filepaths_or_df=filepaths[1:],
        hasher=object_sift,
        max_size=1000,
        min_features=10,
        verbose=True,
    )
    row = pairs[0]
    viz_img = vizualize_pair(
        reference_df.loc[row[0]],
        reference_df.loc[row[1]],
        0.5,
        match_metadata=row[2],
        sanitized=False,
    )
    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/images/debug-image-symmetry-1.png", viz_img)

    # Swap order of ref and query files.
    pairs = ldd.deduplicate(
        filepaths_or_reference_df=filepaths[1:],
        query_filepaths_or_df=filepaths[:1],
        hasher=object_sift,
        max_size=1000,
        min_features=10,
        verbose=True,
    )
    row = pairs[0]
    viz_img = vizualize_pair(
        reference_df.loc[row[0]],
        reference_df.loc[row[1]],
        0.5,
        match_metadata=row[2],
        sanitized=False,
    )
    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tests/images/debug-image-symmetry-2.png", viz_img)
