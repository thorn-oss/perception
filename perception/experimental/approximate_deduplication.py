# pylint: disable=invalid-name

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
        nlist = int(max(min(4 * np.sqrt(ntotal), ntotal / 39), 1))
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


# pylint: disable=too-many-locals
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
            overlap = min([
                len(query_in_match[match]) / length,
                len(match_in_query[match]) / counts[match]
            ])
            if overlap >= minimum_overlap and overlap > 0:
                pairs.append(tuple(sorted([query, match])))
    return list(set(pairs))


def pairs_to_clusters(
        ids: typing.Iterable[str],
        pairs: typing.Iterable[typing.Tuple[str, str]],
        strictness: typing_extensions.
        Literal["clique", "community", "component"] = "clique",
        max_clique_batch_size: int = 1000) -> typing.List[ClusterAssignment]:
    """Given a list of pairs of matching files, compute sets
    of cliques where all files in a clique are connected.
    Args:
        ids: A list of file identifiers (e.g., filepaths).
        pairs: A list of pairs of file identifiers.
        strictness: The level at which groups will be clustered. "component"
            means that all clusters will be connected components. "community"
            will select clusters of files within components that are clustered
            together. "clique" will result in clusters where every file is
            connected to every other file.
        max_clique_batch_size: The maximum batch size for identifying
            cliques.
    Returns:
        A list of cluster assignments (dicts with id and cluster
        entries).
    """
    assert strictness in ["component", "community",
                          "clique"], "Invalid strictness."
    graph = nx.Graph()
    LOGGER.debug("Building graph.")
    graph.add_nodes_from(ids)
    graph.add_edges_from(pairs)
    assignments: typing.List[ClusterAssignment] = []
    cluster_index = 0
    for component in nx.connected_components(graph):
        LOGGER.debug("Got component with size: %s", len(component))
        if strictness == "component":
            assignments.extend([{
                "id": n,
                "cluster": cluster_index
            } for n in component])
            cluster_index += 1
            continue
        for community in nx.algorithms.community.asyn_lpa_communities(
                graph.subgraph(component)):
            LOGGER.debug("Got community with size: %s", len(community))
            if strictness == "community":
                assignments.extend([{
                    "id": n,
                    "cluster": cluster_index
                } for n in community])
                cluster_index += 1
                continue
            community = list(community)  # Need to do this to do batching.
            for start in range(0, len(community), max_clique_batch_size):
                nodes = community[start:start + max_clique_batch_size]
                LOGGER.debug("Creating subgraph with %s nodes.", len(nodes))
                subgraph = graph.subgraph(nodes).copy()
                while subgraph:
                    LOGGER.debug("Subgraph size: %s", len(subgraph))
                    clique = approximation.clique.max_clique(subgraph)
                    assignments.extend([{
                        "id": n,
                        "cluster": cluster_index
                    } for n in clique])
                    cluster_index += 1
                    subgraph.remove_nodes_from(clique)
    return assignments
