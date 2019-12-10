import typing
import warnings
import logging
import uuid
import os

from tqdm import tqdm
import pandas as pd
import imgaug
import cv2

from ..hashers import ImageHasher, tools
from ..tools import deduplicate, flatten
from .common import BenchmarkTransforms, BenchmarkDataset, BenchmarkHashes

# pylint: disable=invalid-name
log = logging.getLogger(__name__)


class BenchmarkImageTransforms(BenchmarkTransforms):
    def compute_hashes(self,
                       hashers: typing.Dict[str, ImageHasher],
                       max_workers: int = 5) -> BenchmarkHashes:
        """Compute hashes for a series of files given some set of hashers.

        Args:
            hashers: A dictionary of hashers.
            max_workers: Maximum number of workers for parallel hash
                computation.

        Returns:
            metrics: A BenchmarkHashes object.
        """
        hashsets = []
        filepaths = self._df['filepath']
        for hasher_name, hasher in hashers.items():
            hash_dicts = hasher.compute_parallel(
                filepaths,
                progress=tqdm,
                progress_desc=f'Computing hashes for {hasher_name}',
                max_workers=max_workers)
            if not hasher.returns_multiple:
                hashes_df = pd.DataFrame.from_records(hash_dicts)
            else:
                hash_groups = [
                    hash_dict['hash']
                    if hash_dict['error'] is None else [None]
                    for hash_dict in hash_dicts
                ]
                hash_group_sizes = [
                    len(hash_group) for hash_group in hash_groups
                ]
                current_hashes = flatten(hash_groups)
                current_filepaths = flatten(
                    [[hash_dict['filepath']] * hash_group_size
                     for hash_dict, hash_group_size in zip(
                         hash_dicts, hash_group_sizes)])
                current_errors = flatten(
                    [[hash_dict['error']] * hash_group_size
                     for hash_dict, hash_group_size in zip(
                         hash_dicts, hash_group_sizes)])
                hashes_df = pd.DataFrame({
                    'error': current_errors,
                    'filepath': current_filepaths,
                    'hash': current_hashes
                })
            hashset = hashes_df.assign(
                hasher_name=hasher_name,
                hasher_hash_length=hasher.hash_length,
                hasher_dtype=hasher.dtype,
                hasher_distance_metric=hasher.distance_metric)
            hashset = hashset.merge(self._df, on='filepath')
            hashsets.append(hashset)
        return BenchmarkHashes(pd.concat(hashsets, sort=True))


class BenchmarkImageDataset(BenchmarkDataset):
    # pylint: disable=too-many-locals
    def deduplicate(self,
                    hasher: ImageHasher,
                    threshold=0.001,
                    isometric=False
                    ) -> typing.Tuple['BenchmarkImageDataset', typing.
                                      Set[typing.Tuple[str, str]]]:
        """ Remove duplicate files from dataset.

        Args:
            files: A list of file paths
            hasher: A hasher to use for finding a duplicate
            threshold: The threshold required for a match
            isometric: Whether to compute the rotated versions of the images

        Returns:
            A list where each entry is a list of files that are
            duplicates of each other. We keep only the last entry.
        """
        pairs: typing.Set[typing.Tuple[str, str]] = set()
        for _, group in tqdm(
                self._df.groupby(['category']),
                desc='Deduplicating categories.'):
            pairs = pairs.union(
                set(
                    deduplicate(
                        files=group['filepath'],
                        hashers=[(hasher, threshold)],
                        isometric=isometric)))
        removed = [pair[0] for pair in pairs]
        return BenchmarkImageDataset(
            self._df[~self._df['filepath'].isin(removed)].copy()), pairs

    def transform(
            self,
            transforms: typing.Dict[str, imgaug.augmenters.meta.Augmenter],
            storage_dir: str,
            errors: str = "raise") -> BenchmarkImageTransforms:
        """Prepare files to be used as part of benchmarking run.

        Args:
            transforms: A dictionary of transformations. The only required
                key is `noop` which determines how the original, untransformed
                image is saved. For a true copy, simply make the `noop` key
                `imgaug.augmenters.Noop()`.
            storage_dir: A directory to store all the images along with
                their transformed counterparts.
            errors: How to handle errors reading files. If "raise", exceptions are
                raised. If "warn", the error is printed as a warning.

        Returns:
            transforms: A BenchmarkImageTransforms object
        """
        assert 'noop' in transforms, 'You must provide a no-op transform such as `lambda img: img`.'

        os.makedirs(storage_dir, exist_ok=True)

        files = self._df.copy()
        files['guid'] = [uuid.uuid4() for n in range(len(files))]

        def apply_transform(files, transform_name):
            transform = transforms[transform_name]
            transformed_arr = []
            for _, row in tqdm(
                    files.iterrows(),
                    desc=f'Creating files for {transform_name}',
                    total=len(files)):
                filepath, guid, category = row[[
                    'filepath', 'guid', 'category'
                ]]
                try:
                    image = tools.read(filepath)
                # pylint: disable=broad-except
                except Exception as exception:
                    message = f'An error occurred reading {filepath}.'
                    if errors == 'raise':
                        raise exception
                    warnings.warn(message, UserWarning)
                    continue
                try:
                    transformed = transform(image=image)
                except Exception:
                    raise Exception(
                        f'An exception occurred while processing {filepath} '
                        f'with transform {transform_name}.')
                transformed_path = os.path.join(
                    storage_dir, f'{guid}_{transform_name}.jpg')
                cv2.imwrite(transformed_path,
                            cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
                transformed_arr.append({
                    'guid': guid,
                    'transform_name': transform_name,
                    'input_filepath': filepath,
                    'filepath': transformed_path,
                    'category': category
                })
            return pd.DataFrame.from_records(transformed_arr)

        results = [apply_transform(files, transform_name='noop')]

        for transform_name in transforms.keys():
            if transform_name == 'noop':
                continue
            results.append(
                apply_transform(results[0], transform_name=transform_name))
        benchmark_transforms = BenchmarkImageTransforms(
            df=pd.concat(results, axis=0, ignore_index=True))
        benchmark_transforms.save(storage_dir)
        return benchmark_transforms
