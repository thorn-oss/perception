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


def test_sift_deduplication():
    tdir = tempfile.TemporaryDirectory()
    watermark = cv2.cvtColor(
        cv2.imread(pt.DEFAULT_TEST_LOGOS[0], cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA)
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, 'test')
               for filepath in pt.DEFAULT_TEST_IMAGES]).transform(
                   transforms={
                       'noop':
                       lambda image: image,
                       'pad':
                       imgaug.augmenters.Pad(percent=0.1),
                       'crop':
                       imgaug.augmenters.Crop(percent=0.1),
                       'watermark':
                       pbit.apply_watermark(watermark, alpha=1, size=0.8)
                   },
                   storage_dir=tdir.name)
    df = transformed._df.set_index('filepath')
    pairs = ldd.deduplicate(
        filepaths_or_reference_df=df.index,
        max_workers=2)  #  Test throws errors if unset.
    clustered = pd.DataFrame(ad.pairs_to_clusters(
        ids=df.index, pairs=pairs)).set_index('id').merge(
            df, left_index=True, right_index=True).reset_index()
    n_clusters = clustered['cluster'].nunique()
    n_transforms = clustered['transform_name'].nunique()
    perfect = clustered.groupby('cluster').apply(
        lambda g: g['guid'].nunique() == 1 and g['transform_name'].nunique() == n_transforms
    ).sum()
    tainted = clustered.groupby('cluster')['guid'].nunique().gt(1).sum()
    pct_perfect = perfect / n_clusters
    pct_tainted = tainted / n_clusters
    assert pct_perfect > 0.1
    assert pct_tainted == 0


def test_sift_deduplication_across_sets():
    tdir = tempfile.TemporaryDirectory()
    watermark = cv2.cvtColor(
        cv2.imread(pt.DEFAULT_TEST_LOGOS[0], cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA)
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, 'test')
               for filepath in pt.DEFAULT_TEST_IMAGES]).transform(
                   transforms={
                       'noop':
                       lambda image: image,
                       'pad':
                       imgaug.augmenters.Pad(percent=0.1),
                       'crop':
                       imgaug.augmenters.Crop(percent=0.1),
                       'watermark':
                       pbit.apply_watermark(watermark, alpha=1, size=0.8)
                   },
                   storage_dir=tdir.name)

    df = transformed._df.set_index('filepath')
    query_images = list(df[df.transform_name == 'noop'].index.values)
    images_to_match_to = list(df[~(df.transform_name == 'noop')].index.values)

    pairs = ldd.deduplicate(
        filepaths_or_reference_df=images_to_match_to,
        query_filepaths_or_df=query_images,
        max_workers=2)  #  Test throws errors if unset.

    assert len(pairs) == 28, "Wrong # of pairs."
    only_one_noop = [
        p for p in pairs if (('noop' in p[0]) != ('noop' in p[1]))
    ]
    assert len(only_one_noop) == len(
        pairs), "All pairs must be between a noop and non-noop file"


def test_validation_for_overlapping_case():
    tdir = tempfile.TemporaryDirectory()
    # Each image will have the center of the other
    # pasted in the top left corner.
    image1 = pht.read(pt.DEFAULT_TEST_IMAGES[0])
    image2 = pht.read(pt.DEFAULT_TEST_IMAGES[1])
    image1[:100, :100] = image2[100:200, 100:200]
    image2[:100, :100] = image1[100:200, 100:200]
    fp1 = os.path.join(tdir.name, 'test1.jpg')
    fp2 = os.path.join(tdir.name, 'test2.jpg')
    cv2.imwrite(fp1, image1[..., ::-1])
    cv2.imwrite(fp2, image2[..., ::-1])
    kp1, des1, dims1 = ldd.generate_image_descriptors(fp1)
    kp2, des2, dims2 = ldd.generate_image_descriptors(fp2)
    # These images should not match.
    assert not ldd.validate_match(
        kp1=kp1, kp2=kp2, des1=des1, des2=des2, dims1=dims1, dims2=dims2)


def test_handling_bad_file_case(caplog):
    tdir = tempfile.TemporaryDirectory()
    missing_file = os.path.join(tdir.name, 'missing-file')
    bad_file_handle = tempfile.NamedTemporaryFile()
    bad_file = bad_file_handle.name
    transformed = pb.BenchmarkImageDataset.from_tuples(
        files=[(filepath, 'test')
               for filepath in pt.DEFAULT_TEST_IMAGES]).transform(
                   transforms={
                       'noop': lambda image: image,
                   },
                   storage_dir=tdir.name)
    df = transformed._df.set_index('filepath')
    df.loc[missing_file] = df.iloc[0]
    df.loc[bad_file] = df.iloc[0]
    pairs = ldd.deduplicate(filepaths_or_reference_df=df.index)
    clustered = pd.DataFrame(ad.pairs_to_clusters(
        ids=df.index, pairs=pairs)).set_index('id').merge(
            df, left_index=True, right_index=True).reset_index()

    assert bad_file not in clustered.index
    assert missing_file not in clustered.index

    bad_file_error = next(
        record for record in caplog.records if bad_file in record.message)
    assert bad_file_error
    assert bad_file_error.levelname == "ERROR"

    missing_file_warning = next(
        record for record in caplog.records if missing_file in record.message)
    assert missing_file_warning
    assert missing_file_warning.levelname == "WARNING"
