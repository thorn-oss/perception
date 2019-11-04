# pylint: disable=protected-access,invalid-name
import shutil
import os

from imgaug import augmenters as iaa
import pytest

from perception import benchmarking, hashers, testing

files = testing.DEFAULT_TEST_IMAGES
dataset = benchmarking.BenchmarkDataset.from_tuples(
    [(fn, i % 2) for i, fn in enumerate(files)])


def test_deduplicate():
    os.makedirs('/tmp/duplicate')
    new_file = '/tmp/duplicate/dup_file.jpg'
    shutil.copy(files[0], new_file)
    duplicated_files = files + [new_file]
    deduplicated, duplicates = benchmarking.BenchmarkDataset.from_tuples(
        [(fn, i % 2) for i, fn in enumerate(duplicated_files)]).deduplicate(
            hasher=hashers.AverageHash(), threshold=1e-2)
    assert len(duplicates) == 1
    assert len(deduplicated._df) == len(files)


def test_bad_dataset():
    bad_files = files + ['tests/images/nonexistent.jpg']
    bad_dataset = benchmarking.BenchmarkDataset.from_tuples(
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
    assert len(dataset._df) == len(files)
    assert len(dataset.filter(category=[0])._df) == len(files) / 2
    with pytest.warns(UserWarning, match='Did not find'):
        assert len(dataset.filter(category=[3])._df) == 0  # pylint: disable=len-as-condition

    dataset.save('/tmp/dataset.zip')
    dataset.save('/tmp/dataset_folder')
    o1 = benchmarking.BenchmarkDataset.load('/tmp/dataset.zip')
    o2 = benchmarking.BenchmarkDataset.load('/tmp/dataset_folder')
    o3 = benchmarking.BenchmarkDataset.load('/tmp/dataset.zip')

    for opened in [o1, o2, o3]:
        assert (opened._df['filepath'].apply(
            os.path.basename) == dataset._df['filepath'].apply(
                os.path.basename)).all()


def test_benchmark_transforms():
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
