from typing import List, Tuple, Dict, Set
from abc import ABC
import warnings
import logging
import zipfile
import uuid
import tempfile
import shutil
import os

import matplotlib.pyplot as plt
from scipy import spatial, stats
from tqdm import tqdm
import pandas as pd
import numpy as np
import imgaug
import cv2

from .hashers.tools import string_to_vector, compute_md5
from .hashers import Hasher
from .tools import deduplicate

# pylint: disable=invalid-name
log = logging.getLogger(__name__)


def compute_threshold_fpr_recall(pos, neg, fpr_threshold=0.001):
    # Sort both arrays according to the positive distance
    neg = neg[pos.argsort()]
    pos = pos[pos.argsort()]

    # Compute false-positive rate for every value in pos
    tp = np.arange(1, len(pos) + 1)
    fp = np.array([(neg <= t).sum() for t in pos])
    fpr = fp / (tp + fp)

    # Choose the optimal threshold
    bad_thresholds = pos[fpr > fpr_threshold]

    # pylint: disable=len-as-condition
    if len(bad_thresholds) > 0:
        optimal_threshold = bad_thresholds[0]
        recovered = (pos < optimal_threshold).sum()
        if recovered == 0:
            optimal_fpr = 0
        else:
            optimal_fpr = fpr[pos < optimal_threshold].max()
        optimal_recall = round(100 * recovered / len(pos), 3)
    else:
        optimal_fpr = 0
        optimal_threshold = pos.max()
        optimal_recall = 100
    return optimal_threshold, optimal_fpr, optimal_recall


class Filterable(ABC):
    _df: pd.DataFrame
    expected_columns: List

    def __init__(self, df):
        # pylint: disable=no-member
        assert sorted(df.columns) == sorted(
            self.expected_columns
        ), f'Column mismatch: Expected {self.expected_columns}, found {df.columns}.'
        # pylint: enable=no-member
        self._df = df

    @property
    def categories(self):
        """The categories included in the dataset"""
        return self._df['category'].unique()

    def filter(self, **kwargs):
        """Obtain a new dataset filtered with the given
        keyword arguments."""
        df = self._df.copy()
        for field, included in kwargs.items():
            existing = self._df[field].unique()
            if not all(inc in existing for inc in included):
                message = 'Did not find {missing} in column {field} dataset.'.format(
                    missing=', '.join(
                        [str(inc) for inc in included if inc not in existing]),
                    field=field)
                warnings.warn(message, UserWarning)
            df = df[df[field].isin(included)]
        return self.__class__(df.copy())


class Saveable(Filterable):
    @classmethod
    def load(cls,
             path_to_zip_or_directory: str,
             storage_dir: str = None,
             verify_md5=True):
        """Load a dataset from a ZIP file or directory.

        Args:
            path_to_zip_or_directory: Pretty self-explanatory
            storage_dir: If providing a ZIP file, where to extract
                the contents. If None, contents will be extracted to
                a folder with the same name as the ZIP file in the
                same directory as the ZIP file.
            verify_md5: Verify md5s when loading
        """

        # Load index whether from inside ZIP file or from directory.
        if os.path.splitext(path_to_zip_or_directory)[1] == '.zip':
            if storage_dir is None:
                storage_dir = os.path.join(
                    os.path.dirname(os.path.abspath(path_to_zip_or_directory)),
                    os.path.splitext(
                        os.path.basename(path_to_zip_or_directory))[0])
                os.makedirs(storage_dir, exist_ok=True)
            with zipfile.ZipFile(path_to_zip_or_directory, 'r') as z:
                # Try extracting only the index at first so we can
                # compare md5.
                z.extract('index.csv', os.path.join(storage_dir))
                index = pd.read_csv(os.path.join(storage_dir, 'index.csv'))
                index['filepath'] = index['filename'].apply(
                    lambda fn: os.path.join(storage_dir, fn))
                if index['filepath'].apply(os.path.isfile).all() and (
                        not verify_md5
                        or all(row['md5'] == compute_md5(row['filepath']))
                        # pylint: disable=bad-continuation
                        for _, row in tqdm(
                            index.iterrows(), desc='Checking cache.')):
                    log.info(
                        'Found all files already extracted. Skipping extraction.'
                    )
                    verify_md5 = False
                else:
                    z.extractall(storage_dir)
        else:
            assert storage_dir is None, 'Storage directory only valid if path is to ZIP file.'
            index = pd.read_csv(
                os.path.join(path_to_zip_or_directory, 'index.csv'))
            index['filepath'] = index['filename'].apply(
                lambda fn: os.path.join(path_to_zip_or_directory, fn))

        if verify_md5:
            assert all(
                row['md5'] == compute_md5(row['filepath']) for _, row in tqdm(
                    index.iterrows(),
                    desc='Performing final md5 integrity check.',
                    total=len(index.index))), 'An md5 mismatch has occurred.'
        return cls(index.drop(['filename', 'md5'], axis=1))

    def save(self, path_to_zip_or_directory):
        """Save a dataset to a directory or ZIP file.

        Args:
            path_to_zip_or_directory: Pretty self-explanatory
        """
        df = self._df
        assert 'filepath' in df.columns, 'Index dataframe must contain md5.'

        # Build index using filename instead of filepath.
        index = df.copy()
        index['filename'] = df['filepath'].apply(os.path.basename)
        if index['filename'].duplicated().sum() > 0:
            warnings.warn(f'Changing filenames to UUID due to duplicates.',
                          UserWarning)

            index['filename'] = [
                str(uuid.uuid4()) + os.path.splitext(row['filename'])[1]
                for _, row in index.iterrows()
            ]
        index['md5'] = [
            compute_md5(filepath)
            for filepath in tqdm(index['filepath'], desc='Computing md5s.')
        ]

        # Add all files as well as the dataframe index to
        # a ZIP file if path is to ZIP file or to the directory if it is
        # not a ZIP file.
        if os.path.splitext(path_to_zip_or_directory)[1] == '.zip':
            with zipfile.ZipFile(path_to_zip_or_directory, 'w') as f:
                with tempfile.TemporaryFile(mode='w+') as index_file:
                    index.drop(
                        'filepath', axis=1).to_csv(
                            index_file, index=False)
                    index_file.seek(0)
                    f.writestr('index.csv', index_file.read())
                for _, row in tqdm(
                        index.iterrows(), desc='Saving files', total=len(df)):
                    f.write(row['filepath'], row['filename'])
        else:
            os.makedirs(path_to_zip_or_directory, exist_ok=True)
            index.drop(
                'filepath', axis=1).to_csv(
                    os.path.join(path_to_zip_or_directory, 'index.csv'),
                    index=False)
            for _, row in tqdm(
                    index.iterrows(), desc='Saving files', total=len(df)):
                if row['filepath'] == os.path.join(path_to_zip_or_directory,
                                                   row['filename']):
                    # The source file is the same as the target file.
                    continue
                shutil.copy(
                    row['filepath'],
                    os.path.join(path_to_zip_or_directory, row['filename']))


class BenchmarkHashes(Filterable):
    """A dataset of hashes for transformed images. It is essentially
    a wrapper around a `pandas.DataFrame` with the following columns:

    - guid
    - filepath
    - category
    - transform_name
    - hasher_name
    - hasher_dtype
    - hasher_distance_metric
    - hasher_hash_length
    """

    expected_columns = [
        'error', 'filepath', 'hash', 'hasher_name', 'hasher_dtype',
        'hasher_distance_metric', 'category', 'guid', 'input_filepath',
        'transform_name', 'hasher_hash_length'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self._metrics: pd.DataFrame = None

    @classmethod
    def load(cls, filepath: str):
        return cls(pd.read_csv(filepath))

    def save(self, filepath):
        self._df.to_csv(filepath, index=False)

    # pylint: disable=too-many-locals
    def compute_metrics(self) -> pd.DataFrame:
        if self._metrics is not None:
            return self._metrics
        metrics = []
        hashsets = self._df
        n_dropped = hashsets['hash'].isnull().sum()
        if n_dropped > 0:
            hashsets = hashsets.dropna(subset=['hash'])
            warnings.warn(f'Dropping {n_dropped} invalid / empty hashes.',
                          UserWarning)
        for (hasher_name, transform_name, category), hashset in tqdm(
                hashsets.groupby(['hasher_name', 'transform_name',
                                  'category']),
                desc='Computing metrics.'):

            # Note the guid filtering below. We need to include only guids
            # for which we have the transform *and* the guid. One of them
            # may have been dropped due to being invalid.
            noops = hashsets[(hashsets['transform_name'] == 'noop')
                             & (hashsets['hasher_name'] == hasher_name)
                             & (hashsets['guid'].isin(hashset['guid']))]

            hashset = hashset[hashset['guid'].isin(noops['guid'])]

            guids_noops = noops.guid.tolist()

            correct_coords = np.arange(0, len(hashset)), hashset.guid.apply(
                guids_noops.index).values

            dtype, distance_metric, hash_length = hashset.iloc[0][[
                'hasher_dtype', 'hasher_distance_metric', 'hasher_hash_length'
            ]]
            distance_matrix = spatial.distance.cdist(
                XA=np.array(
                    hashset.hash.apply(
                        string_to_vector,
                        hash_length=hash_length,
                        dtype=dtype,
                        hash_format='base64').tolist()),
                XB=np.array(
                    noops.hash.apply(
                        string_to_vector,
                        dtype=dtype,
                        hash_format='base64',
                        hash_length=hash_length).tolist()),
                metric=distance_metric)

            closest_guid = noops['guid'].iloc[distance_matrix.argmin(
                axis=1)].values
            distance_to_closest_image = distance_matrix.min(axis=1)
            distance_to_correct_image = distance_matrix[correct_coords]
            rank_of_correct_image = distance_matrix.argsort(axis=1).argsort(
                axis=1)[correct_coords]

            # To compute things for the closest wrong image, we set the
            # distance to the correct image to inf.
            distance_matrix[correct_coords] = np.inf
            distance_to_closest_wrong_image = distance_matrix.min(axis=1)
            closest_wrong_guid = noops['guid'].iloc[distance_matrix.argmin(
                axis=1)].values
            metrics.append(
                pd.DataFrame({
                    'guid':
                    hashset['guid'].values,
                    'transform_name':
                    transform_name,
                    'hasher_name':
                    hasher_name,
                    'category':
                    category,
                    'distance_to_correct_image':
                    distance_to_correct_image,
                    'distance_to_closest_incorrect_image':
                    distance_to_closest_wrong_image,
                    'closest_wrong_guid':
                    closest_wrong_guid,
                    'distance_to_closest_image':
                    distance_to_closest_image,
                    'rank_of_correct_image':
                    rank_of_correct_image,
                    'closest_guid':
                    closest_guid
                }))
        self._metrics = pd.concat(metrics)
        return self._metrics

    # pylint: disable=too-many-locals
    def show_histograms(self, grouping=None, fpr_threshold=0.001):
        """Plot histograms for true and false positives, similar
        to https://tech.okcupid.com/evaluating-perceptual-image-hashes-okcupid/

        Args:
            grouping: List of fields to group by. By default, all fields are used
                (category, and transform_name).
        """
        if grouping is None:
            grouping = ['category', 'transform_name']

        metrics = self.compute_metrics()

        hasher_names = metrics['hasher_name'].unique().tolist()
        bounds = metrics.groupby('hasher_name')[[
            'distance_to_closest_image', 'distance_to_closest_incorrect_image'
        ]].max().max(axis=1)
        if grouping:
            group_names = [
                ':'.join(map(str, row.values))
                for idx, row in metrics[grouping].drop_duplicates().iterrows()
            ]
        else:
            group_names = ['']
        ncols = len(hasher_names)
        nrows = len(group_names)

        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(ncols * 4, nrows * 3),
            sharey=True)

        for group_name, subset in metrics.groupby(['hasher_name'] + grouping):
            # Get names of group and hasher
            if grouping:
                hasher_name = group_name[0]
                group_name = ':'.join(map(str, group_name[1:]))
            else:
                hasher_name = group_name
                group_name = ''

            # Get the correct axis.
            colIdx = hasher_names.index(hasher_name)
            rowIdx = group_names.index(group_name)
            if ncols > 1 and nrows > 1:
                ax = axs[rowIdx, colIdx]
            elif ncols == 1 and nrows == 1:
                ax = axs
            else:
                ax = axs[rowIdx if nrows > 1 else colIdx]

            # Plot the charts
            neg = subset['distance_to_closest_incorrect_image'].values
            pos = subset['distance_to_correct_image'].values
            optimal_threshold, _, optimal_recall = compute_threshold_fpr_recall(
                pos=pos, neg=neg, fpr_threshold=fpr_threshold)
            optimal_threshold = optimal_threshold.round(3)
            emd = stats.wasserstein_distance(pos, neg).round(2)
            ax.hist(neg, label='neg', bins=10)
            ax.hist(pos, label='pos', bins=10)
            ax.text(
                0.5,
                0.5,
                f'Recall: {optimal_recall:.0f}% @ {optimal_threshold}\nemd: {emd:.2f}',
                horizontalalignment='center',
                color='black',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12,
                fontweight=1000)
            ax.set_xlim(-0.05 * bounds[hasher_name], bounds[hasher_name])
            if rowIdx == 0:
                ax.set_title(hasher_name)
                ax.legend()
            if colIdx == 0:
                ax.set_ylabel(group_name)
        fig.tight_layout()

    def compute_threshold_recall(self, fpr_threshold=0.01,
                                 grouping=None) -> pd.DataFrame:
        """Compute a table for threshold and recall for each category, hasher,
        and transformation combinations.

        Args:
            fpr_threshold: The false positive rate threshold to use
                for choosing a distance threshold for each hasher.
            grouping: List of fields to group by. By default, all fields are used
                (category, and transform_name).

        Returns:
            A pandas DataFrame with 7 columns. The key columns are threshold
            (The optimal distance threshold for detecting a match for this
            combination), recall (the number of correct matches divided by
            the number of possible matches), and precision (the number correct
            matches divided by the total number of matches whether correct
            or incorrect).
        """
        if grouping is None:
            grouping = ['category', 'transform_name']

        def group_func(subset):
            neg = subset['distance_to_closest_incorrect_image'].values
            pos = subset['distance_to_correct_image'].values
            optimal_threshold, optimal_fpr, optimal_recall = compute_threshold_fpr_recall(
                pos=pos, neg=neg, fpr_threshold=fpr_threshold)
            return pd.Series({
                'threshold': optimal_threshold,
                'recall': optimal_recall,
                'fpr': optimal_fpr,
                'n_exemplars': len(subset)
            })

        return self.compute_metrics().groupby(
            grouping + ['hasher_name']).apply(group_func)


class BenchmarkTransforms(Saveable):
    """A dataset of transformed images. Essentially wraps a DataFrame with the
    following columns:

    - guid
    - filepath
    - category
    - transform_name
    - input_filepath (for memo purposes only)
    """

    expected_columns = [
        'filepath', 'category', 'transform_name', 'input_filepath', 'guid'
    ]

    def compute_hashes(self, hashers: Dict[str, Hasher],
                       max_workers: int = 5) -> BenchmarkHashes:
        """Compute hashes for a series of files given some set of hashers.

        Args:
            hashers: A dictionary of hashers.
            max_workers: Maximum number of workers for parallel hash
                computation.

        Returns:
            metrics: A dataframe with metrics from the benchmark.
        """
        hashsets = []
        filepaths = self._df['filepath']
        for hasher_name, hasher in hashers.items():
            hashset = pd.DataFrame.from_records(
                hasher.compute_parallel(
                    filepaths,
                    progress=tqdm,
                    progress_desc=f'Computing hashes for {hasher_name}',
                    max_workers=max_workers)).assign(
                        hasher_name=hasher_name,
                        hasher_hash_length=hasher.hash_length,
                        hasher_dtype=hasher.dtype,
                        hasher_distance_metric=hasher.distance_metric)
            hashset = hashset.merge(self._df, on='filepath')
            hashsets.append(hashset)
        return BenchmarkHashes(pd.concat(hashsets))


class BenchmarkDataset(Saveable):
    """A dataset of images separated into
    categories. It is essentially a wrapper around a pandas
    dataframe with the following columns:

    - filepath
    - category
    """

    expected_columns = ['filepath', 'category']

    @classmethod
    def from_tuples(cls, files: List[Tuple[str, str]]):
        """Build dataset from a set of files.

        Args:
            files: A list of tuples where each entry is a pair
                filepath and category.
        """
        df = pd.DataFrame.from_records([{
            'filepath': f,
            'category': c
        } for f, c in files])
        return cls(df)

    # pylint: disable=too-many-locals
    def deduplicate(self, hasher: Hasher, threshold=0.001, isometric=False
                    ) -> Tuple['BenchmarkDataset', Set[Tuple[str, str]]]:
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
        pairs: Set[Tuple[str, str]] = set()
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
        return BenchmarkDataset(
            self._df[~self._df['filepath'].isin(removed)].copy()), pairs

    def transform(self,
                  transforms: Dict[str, imgaug.augmenters.meta.Augmenter],
                  storage_dir: str,
                  errors: str = "raise") -> BenchmarkTransforms:
        """Prepare files to be used as part of benchmarking run.

        Args:
            files: A list of paths to files
            transforms: A dictionary of transformations. The only required
                key is `noop` which determines how the original, untransformed
                image is saved. For a true copy, simply make the `noop` key
                `imgaug.augmenters.Noop()`.
            storage_dir: A directory to store all the images along with
                their transformed counterparts.
            errors: How to handle errors reading files. If "raise", exceptions are
                raised. If "warn", the error is printed as a warning.

        Returns:
            transforms: A BenchmarkTransforms object
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
                image = cv2.imread(filepath)
                if image is None:
                    message = f'An error occurred reading {filepath}.'
                    if errors == 'raise':
                        raise Exception(message)
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
                cv2.imwrite(transformed_path, transformed)
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
        benchmark_transforms = BenchmarkTransforms(
            df=pd.concat(results, axis=0, ignore_index=True))
        benchmark_transforms.save(storage_dir)
        return benchmark_transforms
