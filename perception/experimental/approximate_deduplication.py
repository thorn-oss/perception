# pylint: disable=invalid-name,too-many-locals

import math
import typing
import logging

from networkx.algorithms import approximation
import typing_extensions
import networkx as nx
import numpy as np
import faiss

LOGGER = logging.getLogger(__name__)
DEFAULT_PCT_PROBE = 0

ClusterAssignment = typing_extensions.TypedDict('ClusterAssignment', {
    'cluster': int,
    'id': str
})


def build_index(X: np.ndarray,
                pct_probe: float = DEFAULT_PCT_PROBE,
                approximate=True):
    """Buid a FAISS index from a reference dataframe.

    Args:
        X: The vectors to add to the index.
        pct_probe: The minimum fraction of nearest lists to search. If
            the product of pct_probe and the number of lists is less
            than 1, one list will be searched.
        approximate: Whether to build an approximate or exact index.

    Returns:
        An (index, lookup) tuple where the lookup returns the filepath
        for a given entry in the index.
    """
    if X is None:
        return None
    d = X.shape[1]
    if approximate:
        ntotal = X.shape[0]
        nlist = int(min(4 * np.sqrt(ntotal), ntotal / 39))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        gpu = False
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu = True
        except AttributeError:
            LOGGER.info("Building approximate FAISS index on CPU.")
        index.train(X)
        batch_size = 10_000
        for i in range(0, X.shape[0], batch_size):
            index.add(X[i:i + batch_size])
        if gpu:
            index = faiss.index_gpu_to_cpu(index)  # pylint: disable=no-member
        nprobe = max(math.ceil(pct_probe * nlist), 1)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    else:
        index = faiss.IndexFlat(d)
        index.add(X)  # pylint: disable=no-value-for-parameter
    return index


def compute_euclidean_pairwise_duplicates_approx(X,
                                                 counts,
                                                 threshold,
                                                 minimum_overlap,
                                                 pct_probe=0.1):
    """Provides the same result as perception.extensions.compute_pairwise_duplicates_simple
    but uses an approximate search instead of an exhaustive search, which can dramatically reduce
    processing time.

    Args:
        X: An array of vectors to compute pairs for.
        counts: A list of counts of vectors for separate files in the
            in the vectors (should add up to the length of X)
        threshold: The threshold for a match as a euclidean distance.
        minimum_overlap: The minimum overlap between two files to qualify as a match.
        pct_probe: The minimum percentage of sublists to search for matches. The larger the
            value, the more exhaustive the search.

    Returns:
        A list of pairs of matching file indexes.
    """
    assert counts.sum(
    ) == X.shape[0], "Length of counts incompatible with vectors shape."
    if X.dtype != 'float32':
        # Only make the copy if we have to.
        X = X.astype('float32')
    lookup = []
    for idx, count in enumerate(counts):
        lookup.extend([idx] * count)
    lookup = np.array(lookup)
    index = build_index(X=X, pct_probe=pct_probe, approximate=True)
    pairs = []
    for end, length, query in zip(counts.cumsum(), counts, range(len(counts))):
        if length == 0:
            continue
        Xq = X[end - length:end]
        lims, _, idxs = index.range_search(Xq, threshold**2)
        lims = lims.astype('int32')
        matched = [
            match
            for match in np.unique(lookup[list(set(idxs))])  # type: ignore
            if match != query
        ]
        query_in_match: typing.Mapping[int, set] = {m: set() for m in matched}
        match_in_query: typing.Mapping[int, set] = {m: set() for m in matched}
        for query_idx in range(length):
            for match_idx in idxs[lims[query_idx]:lims[query_idx + 1]]:
                match = lookup[match_idx]
                if match == query:
                    continue
                match_in_query[match].add(match_idx)
                query_in_match[match].add(query_idx)
        for match in matched:
            overlaps = [
                len(query_in_match[match]) / length,
                len(match_in_query[match]) / counts[match]
            ]
            if min(overlaps) > minimum_overlap:
                pairs.append(tuple(sorted([query, match])))
    return list(set(pairs))


def pairs_to_clusters(ids: typing.List[str],
                      pairs: typing.List[typing.Tuple[str, str]]
                      ) -> typing.List[ClusterAssignment]:
    """Given a list of pairs of matching files, compute sets
    of cliques where all files in a clique are connected.

    Args:
        ids: A list of file identifiers (e.g., filepaths).
        pairs: A list of pairs of file identifiers.

    Returns:
        A list of cluster assignments (dicts with id and cluster
        entries).
    """
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from(pairs)
    assignments: typing.List[ClusterAssignment] = []
    cluster_index = 0
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes).copy()
        while subgraph:
            clique = approximation.clique.max_clique(subgraph)
            for entry in clique:
                assignments.append({"id": entry, "cluster": cluster_index})
            subgraph.remove_nodes_from(clique)
            cluster_index += 1
    return assignments