# pylint: disable=protected-access,invalid-name
import tempfile
import shutil
import os

from imgaug import augmenters as iaa
from scipy import spatial
import numpy as np
import pytest

from perception import benchmarking, hashers, testing

files = testing.DEFAULT_TEST_IMAGES
dataset = benchmarking.BenchmarkImageDataset.from_tuples(
    [(fn, i % 2) for i, fn in enumerate(files)])


def test_deduplicate():
    """
    Benchuplicated image to the same.

    Args:
    """
    tempdir = tempfile.TemporaryDirectory()
    new_file = os.path.join(tempdir.name, 'dup_file.jpg')
    shutil.copy(files[0], new_file)
    duplicated_files = files + [new_file]
    deduplicated, duplicates = benchmarking.BenchmarkImageDataset.from_tuples(
        [(fn, i % 2) for i, fn in enumerate(duplicated_files)]).deduplicate(
            hasher=hashers.AverageHash(), threshold=1e-2)
    assert len(duplicates) == 1
    assert len(deduplicated._df) == len(files)


def test_bad_dataset():
    """
    Test for bad bad bad bad bad bad bad datasets.

    Args:
    """
    bad_files = files + ['tests/images/nonexistent.jpg']
    bad_dataset = benchmarking.BenchmarkImageDataset.from_tuples(
        [(fn, i % 2) for i, fn in enumerate(bad_files)])
    transforms = {
        'blur0.05': iaa.GaussianBlur(0.05),
        'noop': iaa.Resize(size=(256, 256))
    }
    with pytest.raises(Exception):
        transformed = bad_dataset.transform(
            transforms=transforms,
            storage_dir='/tmp/transforms',
            errors='raise')
    with pytest.warns(UserWarning, match='occurred reading'):
        transformed = bad_dataset.transform(
            transforms=transforms,
            storage_dir='/tmp/transforms',
            errors='warn')
    assert len(transformed._df) == len(files) * 2


def test_benchmark_dataset():
    """
    Benchmark the dataset as an image isochrone.

    Args:
    """
    assert len(dataset._df) == len(files)
    assert len(dataset.filter(category=[0])._df) == len(files) / 2
    with pytest.warns(UserWarning, match='Did not find'):
        assert len(dataset.filter(category=[3])._df) == 0  # pylint: disable=len-as-condition

    dataset.save('/tmp/dataset.zip')
    dataset.save('/tmp/dataset_folder')
    o1 = benchmarking.BenchmarkImageDataset.load('/tmp/dataset.zip')
    o2 = benchmarking.BenchmarkImageDataset.load('/tmp/dataset_folder')
    o3 = benchmarking.BenchmarkImageDataset.load('/tmp/dataset.zip')

    for opened in [o1, o2, o3]:
        assert (opened._df['filepath'].apply(
            os.path.basename) == dataset._df['filepath'].apply(
                os.path.basename)).all()


def test_benchmark_transforms():
    """
    Benchmark all samples.

    Args:
    """
    transformed = dataset.transform(
        transforms={
            'blur0.05': iaa.GaussianBlur(0.05),
            'noop': iaa.Resize(size=(256, 256))
        },
        storage_dir='/tmp/transforms')

    assert len(transformed._df) == len(files) * 2

    hashes = transformed.compute_hashes(hashers={'pdna': hashers.PHash()})
    tr = hashes.compute_threshold_recall().reset_index()

    hashes._metrics = None
    hashes._df.at[0, 'hash'] = None
    with pytest.warns(UserWarning, match='invalid / empty hashes'):
        hashes.compute_threshold_recall()

    assert (tr[tr['transform_name'] == 'noop']['recall'] == 100.0).all()

    # This is a charting function but we execute it just to make sure
    # it runs without error.
    hashes.show_histograms()


def test_video_benchmark_dataset():
    """
    Benchmark video video.

    Args:
    """
    video_dataset = benchmarking.BenchmarkVideoDataset.from_tuples(
        files=[('perception/testing/videos/v1.m4v',
                'category1'), ('perception/testing/videos/v2.m4v',
                               'category1'),
               ('perception/testing/videos/v1.m4v',
                'category2'), ('perception/testing/videos/v2.m4v',
                               'category2')])
    transforms = {
        'noop':
        benchmarking.video_transforms.get_simple_transform(
            width=128, sar='1/1'),
        'gif':
        benchmarking.video_transforms.get_simple_transform(
            codec='gif', output_ext='.gif'),
        'clip1s':
        benchmarking.video_transforms.get_simple_transform(clip_s=(1, None)),
        'blackpad':
        benchmarking.video_transforms.get_black_frame_padding_transform(
            duration_s=1),
        'slideshow':
        benchmarking.video_transforms.get_slideshow_transform(
            frame_input_rate=1, frame_output_rate=1),
    }
    transformed = video_dataset.transform(
        storage_dir=tempfile.TemporaryDirectory().name, transforms=transforms)
    assert len(transformed._df) == len(transforms) * len(video_dataset._df)
    assert transformed._df['filepath'].isnull().sum() == 0

    # We will compute hashes for each of the transformed
    # videos and check the results for correctness.
    phash_framewise_hasher = hashers.FramewiseHasher(
        frame_hasher=hashers.PHash(),
        interframe_threshold=-1,
        frames_per_second=2)
    hashes = transformed.compute_hashes(
        hashers={'phashframewise': phash_framewise_hasher})

    guid = hashes._df.guid.iloc[0]
    df = hashes._df[hashes._df['guid'] == guid]
    clip1s = df[(df.transform_name == 'clip1s')]
    noop = df[(df.transform_name == 'noop')]
    blackpad = df[(df.transform_name == 'blackpad')]
    slideshow = df[(df.transform_name == 'slideshow')]

    # We should have dropped two hashes from the beginning
    # on the clipped video.
    assert len(clip1s) == len(noop) - 2

    # The first hash from the clipped video should be the
    # same as the third hash from the original
    assert clip1s.hash.iloc[0] == noop.hash.iloc[2]

    # The black padding adds four hashes (two on either side).
    assert len(blackpad) == len(noop) + 4

    # A black frame should yield all zeros for PHash
    assert phash_framewise_hasher.string_to_vector(
        blackpad.iloc[0].hash).sum() == 0

    # The slideshow hashes should be the same as the noop
    # hashes for every other hash.
    assert (noop.hash.values[::2] == slideshow.hash.values[::2]).all()

    # Every second hash in the slideshow should be the same as the
    # previous one.
    for n in range(0, 10, 2):
        assert slideshow.hash.values[n] == slideshow.hash.values[n + 1]


def test_euclidean_extension():
    """
    Computes the cross - correlations.

    Args:
    """

    # This function plainly inplements the process of computing
    # the closest positive and negative examples and their indexes.
    def compute_euclidean_metrics_py(X_noop, X_transformed, mask):
        """
        Compute the hopclidean distance.

        Args:
            X_noop: (todo): write your description
            X_transformed: (bool): write your description
            mask: (array): write your description
        """
        distance_matrix = spatial.distance.cdist(
            XA=X_transformed, XB=X_noop, metric='euclidean')
        pos = np.ma.masked_array(distance_matrix, np.logical_not(mask))
        neg = np.ma.masked_array(distance_matrix, mask)
        distances = np.concatenate([
            neg.min(axis=1).data[np.newaxis],
            pos.min(axis=1).data[np.newaxis]
        ],
                                   axis=0).T
        indexes = np.concatenate(
            [neg.argmin(axis=1)[np.newaxis],
             pos.argmin(axis=1)[np.newaxis]]).T
        return distances, indexes

    X_noop = np.random.uniform(low=0, high=255, size=(5, 144)).astype('int32')
    X_trans = np.random.uniform(
        low=0, high=255, size=(10, 144)).astype('int32')
    mask = np.array([True, False] * 5 * 5).reshape(10, 5)

    distances, indexes = benchmarking.common.extensions.compute_euclidean_metrics(
        X_noop, X_trans, mask)
    distances_py, indexes_py = compute_euclidean_metrics_py(
        X_noop, X_trans, mask)

    assert (indexes_py == indexes).all()
    np.testing.assert_allclose(distances, distances_py)
