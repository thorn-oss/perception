# pylint: disable=invalid-name,line-too-long,no-value-for-parameter,too-many-instance-attributes,too-many-locals,too-many-arguments,no-member
import time
import typing
import warnings
import typing_extensions

import pandas as pd
import numpy as np
import faiss

import perception.hashers.tools as pht

QueryInput = typing_extensions.TypedDict('QueryInput', {
    'id': str,
    'hash': str
})

QueryMatch = typing_extensions.TypedDict('QueryMatch', {
    'id': typing.Any,
    'matches': typing.List[dict]
})


class TuningFailure(Exception):
    pass


class QueryDecodingFailure(Exception):
    pass


def build_query(table, ids, paramstyle, columns):
    query = 'SELECT {} FROM {} WHERE id in {}'
    if paramstyle == 'pyformat':
        sql = query.format(','.join(columns), table, '%(ids)s')
        params = {'ids': tuple(ids)}
    elif paramstyle == 'qmark':
        params = ids
        sql = query.format(','.join(columns), table,
                           '({})'.format(','.join('?' * len(ids))))
    else:
        raise NotImplementedError('Unsupported paramstyle.')
    return sql, params


def query_by_id(con, table, ids, paramstyle,
                extra_columns=None) -> pd.DataFrame:
    """Get data from the database.

    Args:
        con: A connection to the database
        table: The table in which to look up hashes
        ids: The list of IDs to pull
        paramstyle: The paramstyle for the database
        extra_columns: A list of additional (non-ID) columns to pull.
    """
    columns = ['id']
    if extra_columns is not None:
        columns += extra_columns
    if isinstance(ids, np.ndarray):
        # If it's a numpy array, coerce to a list.
        ids = ids.tolist()
    dfs = []
    batch_size = 1000
    for start in range(0, len(ids), batch_size):
        sql, params = build_query(
            table=table,
            ids=ids[start:start + batch_size],
            paramstyle=paramstyle,
            columns=columns)
        dfs.append(pd.read_sql(con=con, sql=sql, params=params))
    return pd.concat(dfs, ignore_index=True).set_index('id')


class ApproximateNearestNeighbors:
    """A wrapper for a FAISS index.

    Args:
        con: A database connection from which to obtain metadata for
            matched hashes.
        table: The table in the database that we should query for metadata.
        paramstyle: The parameter style for the given database
        index: A FAISS index (or filepath to a FAISS index)
        hash_length: The length of the hash that is being matched against.
        metadata_columns: The metadata that should be returned for queries.
        dtype: The data type for the vectors
        distance_metric: The distance metric for the vectors
    """

    def __init__(
            self,
            con,
            table,
            paramstyle,
            index,
            hash_length,
            metadata_columns=None,
            dtype='uint8',
            distance_metric='euclidean',
    ):
        assert dtype == 'uint8', 'Only unsigned 8-bit integer hashes are supported at this time.'
        assert distance_metric == 'euclidean', 'Only euclidean distance is supported at this time.'
        if isinstance(index, str):
            index = faiss.read_index(index)
        self.con = con
        self.index = index
        self.distance_metric = distance_metric
        self.hash_length = hash_length
        self.dtype = dtype
        self.table = table
        self.metadata_columns = metadata_columns
        self.paramstyle = paramstyle
        assert self.index.d == self.hash_length, 'Index is incompatible with hash length.'

    @classmethod
    def from_database(cls,
                      con,
                      table,
                      paramstyle,
                      hash_length,
                      ids_train=None,
                      train_size=None,
                      chunksize=100000,
                      metadata_columns=None,
                      index=None,
                      gpu=False,
                      dtype='uint8',
                      distance_metric='euclidean'):
        """Train and build a FAISS index from a database connection.

        Args:
            con: A database connection from which to obtain metadata for
                matched hashes.
            table: The table in the database that we should query for metadata.
            paramstyle: The parameter style for the given database
            hash_length: The length of the hash that is being matched against.
            ids_train: The IDs for the vectors to train on.
            train_size: The number of vectors to use for training. Will be
                randomly selected from 1 to the number of vectors in the database.
                Ignored if ids_train is not None.
            chunksize: The chunks of data to draw from the database at a time
                when adding vectors to the index.
            metadata_columns: The metadata that should be returned for queries.
            index: If a pretrained index is provided, training will be skipped,
                any existing vectors will be discarded, and the index will be
                repopulated with the current contents of the database.
            gpu: If true, will attempt to carry out training on a GPU.
            dtype: The data type for the vectors
            distance_metric: The distance metric for the vectors
        """
        assert dtype == 'uint8', 'Only unsigned 8-bit integer hashes are supported at this time.'
        assert distance_metric == 'euclidean', 'Only euclidean distance is supported at this time.'
        if index is None:
            # Train the index using the practices from
            # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-below-1m-vectors-ivfx
            ntotal = pd.read_sql(
                sql="select count(*) as count from hashes",
                con=con).iloc[0]['count']
            assert train_size <= ntotal, 'Cannot train on more hashes than are available.'
            nlist = int(min(4 * np.sqrt(ntotal), ntotal / 39))
            min_train_size = 39 * nlist
            if ids_train is not None:
                train_size = len(ids_train)
            if train_size is None:
                train_size = min_train_size
            assert train_size >= min_train_size, f'Training an index used for {ntotal} hashes requires at least {min_train_size} training hashes.'
            if ids_train is None:
                ids_train = np.random.choice(
                    np.arange(ntotal), size=train_size, replace=False)
            df_train = query_by_id(
                con=con,
                table=table,
                ids=ids_train,
                paramstyle=paramstyle,
                extra_columns=['hash'])
            x_train = np.array([
                np.frombuffer(h, dtype=dtype) for h in df_train['hash']
            ]).astype('float32')
            assert x_train.shape[
                1] == hash_length, 'Hashes are of incorrect length.'

            index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(hash_length), hash_length, nlist)
            if gpu:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.train(x_train)
                index = faiss.index_gpu_to_cpu(gpu_index)
            else:
                index.train(x_train)
        else:
            index.reset()

        # Add hashes to the index in chunks.
        for df_add in pd.read_sql(
                sql=f"SELECT id, hash FROM {table}", con=con,
                chunksize=chunksize):
            x_add = np.array([
                np.frombuffer(h, dtype=dtype) for h in df_add['hash']
            ]).astype('float32')
            index.add_with_ids(x_add, df_add['id'].values)
        return cls(
            con=con,
            index=index,
            hash_length=hash_length,
            distance_metric=distance_metric,
            dtype=dtype,
            table=table,
            paramstyle=paramstyle,
            metadata_columns=metadata_columns)

    def query_by_id(self, ids, include_metadata=True,
                    include_hashes=False) -> pd.DataFrame:
        """Get data from the database.

        Args:
            ids: The hash IDs to get from the database.
            include_metadata: Whether to include metadata columns.
            include_hashes: Whether to include the hashes
        """
        if not self.metadata_columns and include_metadata and not include_hashes:
            # There won't be anything to  return.
            return None
        extra_columns = []
        if self.metadata_columns and include_metadata:
            extra_columns += self.metadata_columns
        if include_hashes:
            extra_columns += ['hash']
        return query_by_id(
            con=self.con,
            table=self.table,
            ids=ids,
            paramstyle=self.paramstyle,
            extra_columns=extra_columns)

    def string_to_vector(self, s: str, hash_format='base64') -> np.ndarray:
        """Convert a string to vector form.

        Args:
            s: The hash string
            hash_format: The format for the hash string
        """
        return pht.string_to_vector(
            s,
            hash_format=hash_format,
            dtype=self.dtype,
            hash_length=self.hash_length)

    def vector_to_string(self, vector, hash_format='base64') -> str:
        """Convert a vector back to string

        Args:
            vector: The hash vector
            hash_format: The format for the hash
        """

        return pht.vector_to_string(
            vector, dtype=self.dtype, hash_format=hash_format)

    def search(self,
               queries: typing.List[QueryInput],
               threshold: int = None,
               threshold_func: typing.Callable[[np.ndarray], int] = None,
               hash_format='base64',
               k=1):
        """Search the index and return matches.

        Args:
            queries: A list of queries in the form of {"id": <id>, "hash": "<hash_string>"}
            threshold: The threshold to use for matching. Takes precedence over threshold_func.
            threshold_func: A function that, given a query vector, returns the desired match threshold for that query.
            hash_format: The hash format used for the strings in the query.
            k: The number of nearest neighbors to return.

        Returns:
            Matches in the form of a list of dicts of the form:
            { "id": <query ID>, "matches": [{"distance": <distance>, "id": <match ID>, "metadata": {}}]}

            The metadata consists of the contents of the metadata columns specified for this matching
            instance.
        """
        try:
            xq = np.array([
                self.string_to_vector(h['hash'], hash_format=hash_format)
                for h in queries
            ]).astype('float32')
        except Exception as exc:
            raise QueryDecodingFailure('Failed to parse hash query.') from exc
        if threshold:
            thresholds = np.ones((len(xq), 1)) * threshold
        if not threshold and threshold_func:
            thresholds = threshold_func(xq)
        else:
            thresholds = np.ones((len(xq), 1)) * np.inf
        distances, indices = self.index.search(xq, k=k)
        distances = np.sqrt(distances)
        metadata = None if not self.metadata_columns else self.query_by_id(
            ids=np.unique(indices[distances < thresholds]))
        matches: typing.List[QueryMatch] = []
        for match_distances, match_ids, q, q_threshold in zip(
                distances, indices, queries, thresholds):
            match_filter = match_distances < q_threshold
            match_ids = match_ids[match_filter]
            match_distances = match_distances[match_filter]
            match: QueryMatch = {'id': q['id'], 'matches': []}
            for match_id, distance in zip(match_ids, match_distances):
                entry = {'distance': float(distance), 'id': match_id}
                if metadata is not None:
                    entry['metadata'] = metadata.loc[match_id].to_dict()
                match['matches'].append(entry)
            matches.append(match)
        return matches

    def tune(self, n_query=100, min_recall=99, max_noise=3):
        """Obtain minimum value for nprobe that achieves a target level of recall.
        Args:
            n_query: The number of hashes to use as test hashes.
            min_recall: The minimum desired recall for the index.
            max_noise: The maximum amount of noise to add to each test hash

        Returns:
            A tuple of recall, latency (in ms), and nprobe where the nprobe
            value is the one that achieved the resulting recall.

        Raises:
            TuningFailure if no suitable nprobe value is found.
        """
        assert n_query <= self.ntotal, 'Cannot use a test larger than ntotal (total number of hashes).'

        # Pick a random set of query hashes
        ids = np.random.choice(
            np.arange(1, self.ntotal + 1), size=n_query, replace=False)
        df = self.query_by_id(ids, include_metadata=False, include_hashes=True)
        xq = np.uint8([np.frombuffer(v, dtype=self.dtype) for v in df['hash']])

        noise = np.random.randint(
            low=(-xq.astype('int32')).clip(-max_noise, max_noise),
            high=(255 - xq.astype('float32')).clip(-max_noise, max_noise))
        xq = (xq.astype('int32') + noise).astype('uint8').astype('float32')

        if min_recall == 100:
            warnings.warn(
                '100% recall can only be ensured with exhaustive search.',
                UserWarning)
            self.set_nprobe(self.nlist)
            start = time.time()
            self.index.search(xq, k=1)
            latency = time.time() - start
            return (100, 1000 * latency, self.nlist)

        # Make the search exhaustive so we get ground truth.
        self.set_nprobe(self.nlist)
        _, expected = self.index.search(xq, k=1)

        for nprobe in range(1, self.nlist):
            self.set_nprobe(nprobe)
            start = time.time()
            _, actual = self.index.search(xq, k=1)
            latency = time.time() - start
            recall = 100 * (actual[:, 0] == expected).sum() / xq.shape[0]
            if recall >= min_recall:
                break
        else:
            # If we never break, it means we never reached the target recall
            # for this query.
            raise TuningFailure(
                'Failed to find suitable parameters for selected recall.')
        return recall, 1000 * latency, nprobe

    def save(self, filepath):
        """Save an index to disk.

        Args:
            filepath: Where to save the index.
        """
        faiss.write_index(self.index, filepath)

    def set_nprobe(self, nprobe) -> int:
        """Set the value of nprobe.

        Args:
            nprobe: The new value for nprobe
        """
        faiss.ParameterSpace().set_index_parameter(self.index, "nprobe",
                                                   nprobe)
        return faiss.downcast_index(self.index).nprobe

    @property
    def nlist(self):
        """The number of lists in the index."""
        return faiss.downcast_index(self.index).nlist

    @property
    def nprobe(self):
        """The current value of nprobe."""
        return faiss.downcast_index(self.index).nprobe

    @property
    def ntotal(self):
        """The number of vectors in the index."""
        return self.index.ntotal
