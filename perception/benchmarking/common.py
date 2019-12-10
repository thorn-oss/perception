# pylint: disable=invalid-name
import concurrent.futures
from abc import ABC
import logging
import tempfile
import typing
import itertools
import warnings
import zipfile
import shutil
import uuid
import os

import matplotlib.pyplot as plt
from scipy import spatial, stats
import pandas as pd
import numpy as np
import tqdm

from ..hashers.tools import compute_md5, string_to_vector

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
    expected_columns: typing.List

    def __init__(self, df):
        # pylint: disable=no-member
        assert sorted(df.columns) == sorted(
            self.expected_columns
        ), f'Column mismatch: Expected {sorted(self.expected_columns)}, found {sorted(df.columns)}.'
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
                        for _, row in tqdm.tqdm(
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
                row['md5'] == compute_md5(row['filepath'])
                for _, row in tqdm.tqdm(
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
        index['filename'] = df['filepath'].apply(
            lambda filepath: os.path.basename(filepath) if filepath is not None else filepath
        )
        if index['filename'].dropna().duplicated().sum() > 0:
            warnings.warn(f'Changing filenames to UUID due to duplicates.',
                          UserWarning)

            index['filename'] = [
                str(uuid.uuid4()) + os.path.splitext(row['filename'])[1]
                if row['filename'] is not None else None
                for _, row in index.iterrows()
            ]
        index['md5'] = [
            compute_md5(filepath) if filepath is not None else None
            for filepath in tqdm.tqdm(
                index['filepath'], desc='Computing md5s.')
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
                for _, row in tqdm.tqdm(
                        index.iterrows(), desc='Saving files', total=len(df)):
                    if row['filepath'] is None:
                        #  There was an error associated with this file.
                        continue
                    f.write(row['filepath'], row['filename'])
        else:
            os.makedirs(path_to_zip_or_directory, exist_ok=True)
            index.drop(
                'filepath', axis=1).to_csv(
                    os.path.join(path_to_zip_or_directory, 'index.csv'),
                    index=False)
            for _, row in tqdm.tqdm(
                    index.iterrows(), desc='Saving files', total=len(df)):
                if row['filepath'] is None:
                    # There was an error associated with this file.
                    continue
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
    - error
    - filepath
    - category
    - transform_name
    - hasher_name
    - hasher_dtype
    - hasher_distance_metric
    - hasher_hash_length
    - hash
    """

    expected_columns = [
        'error', 'filepath', 'hash', 'hasher_name', 'hasher_dtype',
        'hasher_distance_metric', 'category', 'guid', 'input_filepath',
        'transform_name', 'hasher_hash_length'
    ]

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self._metrics: pd.DataFrame = None

    def __add__(self, other):
        return BenchmarkHashes(
            df=pd.concat([self._df, other._df]).drop_duplicates())

    def __radd__(self, other):
        return self.__add__(other)

    @classmethod
    def load(cls, filepath: str):
        return cls(pd.read_csv(filepath))

    def save(self, filepath):
        self._df.to_csv(filepath, index=False)

    # pylint: disable=too-many-locals
    def compute_metrics(self,
                        custom_distance_metrics: dict = None) -> pd.DataFrame:
        if self._metrics is not None:
            return self._metrics
        metrics = []
        hashsets = self._df.sort_values('guid')
        n_dropped = hashsets['hash'].isnull().sum()
        if n_dropped > 0:
            hashsets = hashsets.dropna(subset=['hash'])
            warnings.warn(f'Dropping {n_dropped} invalid / empty hashes.',
                          UserWarning)
        for (hasher_name, transform_name, category), hashset in tqdm.tqdm(
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
            dtype, distance_metric, hash_length = hashset.iloc[0][[
                'hasher_dtype', 'hasher_distance_metric', 'hasher_hash_length'
            ]]
            n_noops = len(noops.guid)
            n_hashset = len(hashset.guid)
            if distance_metric != 'custom':
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
            else:
                assert (
                    custom_distance_metrics is not None and
                    hasher_name in custom_distance_metrics
                ), \
                    f'You must provide a custom distance metric for {hasher_name}.'
                noops_hash_values = noops.hash.values
                hashset_hash_values = hashset.hash.values
                distance_matrix = np.zeros((n_hashset, n_noops))
                distance_function = custom_distance_metrics[hasher_name]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures_to_indices = {
                        executor.submit(distance_function,
                                        hashset_hash_values[i1],
                                        noops_hash_values[i2]): (i1, i2)
                        for i1, i2 in itertools.product(
                            range(n_hashset), range(n_noops))
                    }
                    for future in concurrent.futures.as_completed(
                            futures_to_indices):
                        i1, i2 = futures_to_indices[future]
                        distance_matrix[i1, i2] = future.result()
            distance_to_closest_image = distance_matrix.min(axis=1)
            previous_guid = None
            start = None
            end = 0
            mask = np.zeros_like(distance_matrix, dtype='bool')
            noop_guids = noops.guid.values
            for current_guid, row in zip(hashset.guid, mask):
                if previous_guid is None or current_guid != previous_guid:
                    start = end
                    end = start + next(
                        (other_index
                         for other_index, guid in enumerate(noop_guids[start:])
                         if guid != current_guid), n_noops)
                    previous_guid = current_guid
                row[start:end] = True
            distance_matrix_correct_image = np.ma.masked_array(
                distance_matrix, np.logical_not(mask))
            distance_matrix_incorrect_image = np.ma.masked_array(
                distance_matrix, mask)
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
                    'distance_to_closest_correct_image':
                    distance_matrix_correct_image.min(axis=1),
                    'distance_to_closest_incorrect_image':
                    distance_matrix_incorrect_image.min(axis=1),
                    'distance_to_closest_image':
                    distance_to_closest_image,
                    'closest_incorrect_guid':
                    noop_guids[distance_matrix_incorrect_image.argmin(axis=1)]
                }))
        metrics = pd.concat(metrics)
        self._metrics = metrics
        return metrics

    # pylint: disable=too-many-locals
    def show_histograms(self, grouping=None, fpr_threshold=0.001, **kwargs):
        """Plot histograms for true and false positives, similar
        to https://tech.okcupid.com/evaluating-perceptual-image-hashes-okcupid/
        Additional arguments passed to compute_metrics.

        Args:
            grouping: List of fields to group by. By default, all fields are used
                (category, and transform_name).
        """
        if grouping is None:
            grouping = ['category', 'transform_name']

        metrics = self.compute_metrics(**kwargs)

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
            pos = subset['distance_to_closest_correct_image'].values
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

    def compute_threshold_recall(self,
                                 fpr_threshold=0.01,
                                 grouping=None,
                                 **kwargs) -> pd.DataFrame:
        """Compute a table for threshold and recall for each category, hasher,
        and transformation combinations. Additional arguments passed to compute_metrics.

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
            neg = subset.groupby(
                'guid')['distance_to_closest_incorrect_image'].min().values
            pos = subset.groupby(
                'guid')['distance_to_closest_correct_image'].min().values
            optimal_threshold, optimal_fpr, optimal_recall = compute_threshold_fpr_recall(
                pos=pos, neg=neg, fpr_threshold=fpr_threshold)
            return pd.Series({
                'threshold': optimal_threshold,
                'recall': optimal_recall,
                'fpr': optimal_fpr,
                'n_exemplars': len(subset)
            })

        return self.compute_metrics(
            **kwargs).groupby(grouping + ['hasher_name']).apply(group_func)


class BenchmarkDataset(Saveable):
    """A dataset of images separated into
    categories. It is essentially a wrapper around a pandas
    dataframe with the following columns:

    - filepath
    - category
    """

    expected_columns = ['filepath', 'category']

    @classmethod
    def from_tuples(cls, files: typing.List[typing.Tuple[str, str]]):
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

    def transform(self, transforms, storage_dir, errors):
        raise NotImplementedError()


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

    def compute_hashes(self, hashers, max_workers):
        raise NotImplementedError()
