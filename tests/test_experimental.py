# pylint: disable=protected-access,invalid-name
import os
import tempfile
import imgaug
import cv2
import pandas as pd
import perception.testing as pt
import perception.benchmarking as pb
import perception.hashers.tools as pht
import perception.benchmarking.image_transforms as pbit
import perception.experimental.local_descriptor_deduplication as ldd
import perception.experimental.approximate_deduplication as ad
import pytest


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
