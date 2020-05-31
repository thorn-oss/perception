import concurrent.futures
import typing
import uuid
import os

import tqdm
import pandas as pd

from ..tools import flatten
from ..hashers import VideoHasher, tools
from .common import BenchmarkDataset, BenchmarkTransforms, BenchmarkHashes


def _process_row(row, hashers, framerates):
    error = None
    try:
        assert not pd.isnull(row['filepath']), 'No filepath provided.'
        hashes = tools.compute_synchronized_video_hashes(
            filepath=row['filepath'],
            hashers=hashers,
            framerates=framerates,
            hash_format='base64')
    # pylint: disable=broad-except
    except Exception as exception:
        error = str(exception)
        hashes = {
            hasher_name: [None] if hasher.returns_multiple else None
            for hasher_name, hasher in hashers.items()
        }
    base_dict = {
        'guid': row['guid'],
        'filepath': row['filepath'],
        'error': error,
        'category': row['category'],
        'transform_name': row['transform_name'],
        'input_filepath': row['input_filepath']
    }
    hash_dicts = []
    for hasher_name, hasher in hashers.items():
        base_hash_dict = {
            'hasher_name': hasher_name,
            'hasher_dtype': hasher.dtype,
            'hasher_distance_metric': hasher.distance_metric,
            'hasher_hash_length': hasher.hash_length,
        }
        if not hasher.returns_multiple:
            hash_dicts.append({
                **{
                    'hash': hashes[hasher_name],
                },
                **base_hash_dict
            })
        else:
            for hash_value in hashes[hasher_name]:
                hash_dicts.append({
                    **{
                        'hash': hash_value,
                    },
                    **base_hash_dict
                })
    return [{**hash_dict, **base_dict} for hash_dict in hash_dicts]


class BenchmarkVideoDataset(BenchmarkDataset):
    def transform(self,
                  transforms: typing.Dict[str, typing.Callable],
                  storage_dir: str,
                  errors: str = "raise"):
        """Prepare files to be used as part of benchmarking run.

        Args:
            transforms: A dictionary of transformations. The only required
                key is `noop` which determines how the original, untransformed
                video is saved. Each transform should be a callable function with
                that accepts an `input_filepath` and `output_filepath` argument and
                it should return the `output_filepath` (which may have a different
                extension appended by the transform function).
            storage_dir: A directory to store all the videos along with
                their transformed counterparts.
            errors: How to handle errors reading files. If "raise", exceptions are
                raised. If "warn", the error is printed as a warning.

        Returns:
            transforms: A BenchmarkVideoTransforms object
        """
        assert 'noop' in transforms, 'You must provide a no-op transform.'

        os.makedirs(storage_dir, exist_ok=True)

        files = self._df.copy()
        files['guid'] = [uuid.uuid4() for n in range(len(files))]

        def apply_transform_to_file(input_filepath, guid, transform_name,
                                    category):
            if input_filepath is None:
                # This can happen if the noop transform did not yield
                # a file. We don't want to drop the records so we
                # keep them.
                return {
                    'guid': guid,
                    'error': 'No source file provided',
                    'transform_name': transform_name,
                    'input_filepath': input_filepath,
                    'filepath': None,
                    'category': category
                }
            try:
                output_filepath = transforms[transform_name](
                    input_filepath,
                    output_filepath=os.path.join(storage_dir,
                                                 f'{guid}_{transform_name}'))
                error = None
            # pylint: disable=invalid-name,broad-except
            except Exception as e:
                output_filepath = None
                error = str(e)
            return {
                'guid': guid,
                'error': error,
                'transform_name': transform_name,
                'input_filepath': input_filepath,
                'filepath': output_filepath,
                'category': category
            }

        def apply_transform_to_files(files, transform_name):
            return pd.DataFrame.from_records([
                apply_transform_to_file(
                    input_filepath=row['filepath'],
                    guid=row['guid'],
                    transform_name=transform_name,
                    category=row['category']) for _, row in tqdm.tqdm(
                        files.iterrows(),
                        desc=f'Creating files for {transform_name}',
                        total=len(files))
            ])

        results = [apply_transform_to_files(files, transform_name='noop')]
        for transform_name in transforms.keys():
            if transform_name == 'noop':
                continue
            results.append(
                apply_transform_to_files(
                    results[0], transform_name=transform_name))
        benchmark_transforms = BenchmarkVideoTransforms(
            df=pd.concat(results, axis=0, ignore_index=True))
        benchmark_transforms.save(storage_dir)
        return benchmark_transforms


class BenchmarkVideoTransforms(BenchmarkTransforms):
    expected_columns = [
        'filepath', 'category', 'transform_name', 'input_filepath', 'guid',
        'error'
    ]

    def compute_hashes(self,
                       hashers: typing.Dict[str, VideoHasher],
                       max_workers: int = 5) -> BenchmarkHashes:
        """Compute hashes for a series of files given some set of hashers.

        Args:
            hashers: A dictionary of hashers.
            max_workers: Maximum number of workers for parallel hash
                computation.

        Returns:
            hashes: A BenchmarkHashes object.
        """
        id_rates = {
            hasher_name: hasher.frames_per_second
            for hasher_name, hasher in hashers.items()
            if hasher.frames_per_second is not None
        }
        if id_rates:
            framerates = tools.get_common_framerates({
                hasher_name: hasher.frames_per_second
                for hasher_name, hasher in hashers.items()
                if hasher.frames_per_second is not None
            })
        else:
            framerates = {}

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_row,
                    row=row,
                    framerates=framerates,
                    hashers=hashers) for index, row in self._df.iterrows()
            ]
            return BenchmarkHashes(
                pd.DataFrame.from_records(
                    flatten([
                        future.result() for future in tqdm.tqdm(
                            concurrent.futures.as_completed(futures),
                            desc='Computing hashes.',
                            total=len(self._df))
                    ])))
