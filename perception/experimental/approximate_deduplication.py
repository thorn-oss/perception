import logging
import math
import os.path as op
import typing

import faiss
import networkit as nk
import numpy as np
import tqdm
import typing_extensions

LOGGER = logging.getLogger(__name__)
DEFAULT_PCT_PROBE = 0


# For faiss training on datasets larger than 50,000 vectors, we take a random sub-sample.
TRAIN_LARGE_SIZE: int = 50_000


class ClusterAssignment(typing_extensions.TypedDict):
    cluster: int
    id: typing.Any


def build_index(
    X: np.ndarray,
    pct_probe: float = DEFAULT_PCT_PROBE,
    approximate: bool = True,
    use_gpu: bool = True,
):
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
    X = X.astype("float32")
    d = X.shape[1]
    if approximate:
        ntotal = X.shape[0]
        nlist = int(max(min(4 * np.sqrt(ntotal), ntotal / 39), 1))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        gpu = False
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu = True
            except AttributeError:
                LOGGER.info("Building approximate FAISS index on CPU.")

        if X.shape[0] > TRAIN_LARGE_SIZE:
            # Take random sample of 50,000 or 39 points per centroid.
            # 39 points per centroid is the min for for not getting warnings.
            # https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
            sample_size = max(39 * nlist, TRAIN_LARGE_SIZE)
            index.train(X[np.random.choice(X.shape[0], sample_size, replace=False)])
        else:
            index.train(X)

        batch_size = 10_000
        for i in range(0, X.shape[0], batch_size):
            index.add(X[i : i + batch_size])
        if gpu:
            index = faiss.index_gpu_to_cpu(index)
        nprobe = max(math.ceil(pct_probe * nlist), 1)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
    else:
        index = faiss.IndexFlat(d)
        index.add(X)
    return index


def compute_euclidean_pairwise_duplicates_approx(
    X,
    counts,
    threshold,
    minimum_overlap,
    Y=None,
    y_counts=None,
    pct_probe=0.1,
    use_gpu: bool = True,
    faiss_cache_path: str | None = None,
    show_progress: bool = False,
):
    """Provides the same result as perception.extensions.compute_pairwise_duplicates_simple
    but uses an approximate search instead of an exhaustive search, which can dramatically reduce
    processing time.

    Args:
        X: An array of vectors to compute pairs for.
        Y: if provided we search in X for Y vectors.
        counts: A list of counts of vectors for separate files in the
            in the vectors (should add up to the length of X)
        threshold: The threshold for a match as a euclidean distance.
        minimum_overlap: The minimum overlap between two files to qualify as a match.
        pct_probe: The minimum percentage of sublists to search for matches. The larger the
            value, the more exhaustive the search.
        faiss_cache_path: If provided load any existing faiss index from this path, and if
            it does not exist then save the generated faiss index to the path.
        show_progress: Whether or not to show a progress bar while computing pairs
    Returns:
        A list of pairs of matching file indexes.
    """
    assert (
        counts.sum() == X.shape[0]
    ), "Length of counts incompatible with vectors shape."
    assert (Y is None) == (
        y_counts is None
    ), "Must provide both or neither for y, y_counts."
    if X.dtype != "float32":
        # Only make the copy if we have to.
        X = X.astype("float32")

    if Y is not None and Y.dtype != "float32":
        # Only make the copy if we have to.
        Y = Y.astype("float32")

    lookup_ = []
    for idx, count in enumerate(counts):
        lookup_.extend([idx] * count)
    lookup = np.array(lookup_)

    if faiss_cache_path is not None and op.exists(faiss_cache_path):
        LOGGER.debug("Loading cached FAISS index from %s", faiss_cache_path)
        index = faiss.read_index(faiss_cache_path)
        assert (
            X.shape[0] == index.ntotal
        ), "Cached FAISS index does not match provided X."
    else:
        LOGGER.debug("Building FAISS index.")
        index = build_index(X=X, pct_probe=pct_probe, approximate=True, use_gpu=use_gpu)
        if faiss_cache_path is not None:
            faiss.write_index(index, faiss_cache_path)

    LOGGER.debug("FAISS index ready, start aprox search")
    pairs = []

    # Only use y_counts if present.
    if y_counts is None:
        iterator_counts = counts
        M = X
    else:
        iterator_counts = y_counts
        M = Y

    for end, length, query in tqdm.tqdm(
        zip(iterator_counts.cumsum(), iterator_counts, range(len(iterator_counts))),
        total=len(iterator_counts),
        disable=not show_progress,
        desc="Vectors",
    ):
        if length == 0:
            continue
        Xq = M[end - length : end]
        lims, _, idxs = index.range_search(Xq, threshold**2)
        lims = lims.astype("int32")
        matched = [
            match
            for match in np.unique(lookup[list(set(idxs))])  # type: ignore
            if match != query
            or Y is not None  # Protect self matches if Y is not present.
        ]
        query_in_match: typing.Mapping[int, set] = {m: set() for m in matched}
        match_in_query: typing.Mapping[int, set] = {m: set() for m in matched}
        for query_idx in range(length):
            for match_idx in idxs[lims[query_idx] : lims[query_idx + 1]]:
                match = lookup[match_idx]
                if (
                    match == query and Y is None
                ):  # Protect self matches if Y is not present.
                    continue
                match_in_query[match].add(match_idx)
                query_in_match[match].add(query_idx)
        for match in matched:
            overlap = min(
                [
                    len(query_in_match[match]) / length,
                    len(match_in_query[match]) / counts[match],
                ]
            )
            if overlap >= minimum_overlap and overlap > 0:
                if Y is None:
                    pairs.append(tuple(sorted([query, match])))
                else:
                    pairs.append(tuple([query, match]))
    return list(set(pairs))


def pairs_to_clusters(
    ids: typing.Iterable[str],
    pairs: typing.Iterable[tuple[str, str]],
    strictness: typing_extensions.Literal[
        "clique", "community", "component"
    ] = "clique",
    max_clique_batch_size: int = 1000,
) -> list[ClusterAssignment]:
    """Given a list of pairs of matching files, compute sets
    of cliques where all files in a clique are connected.
    Args:
        ids: A list of node ids (e.g., filepaths).
        pairs: A list of pairs of node ids, each pair is assumed to have an edge
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
    assert strictness in ["component", "community", "clique"], "Invalid strictness."
    list_ids = list(ids)
    id_to_node_map = {v: i for i, v in enumerate(list_ids)}
    node_to_id_map = {v: k for k, v in id_to_node_map.items()}

    LOGGER.debug("Building graph.")
    graph = nk.Graph(len(list_ids))
    node_pairs = {(id_to_node_map[pair[0]], id_to_node_map[pair[1]]) for pair in pairs}
    for node_pair in node_pairs:
        graph.addEdge(node_pair[0], node_pair[1])

    assignments: list[ClusterAssignment] = []
    cluster_index = 0
    cc_query = nk.components.ConnectedComponents(graph)
    cc_query.run()
    components = cc_query.getComponents()

    for component in components:
        LOGGER.debug("Got component with size: %s", len(component))
        if strictness == "component":
            assignments.extend(
                [{"id": node_to_id_map[n], "cluster": cluster_index} for n in component]
            )
            cluster_index += 1
            continue
        # Map between node values for a connected component
        component_node_map = dict(enumerate(component))
        cc_sub_graph = nk.graphtools.subgraphFromNodes(graph, component, compact=True)
        algo = nk.community.PLP(cc_sub_graph)
        algo.run()
        communities = algo.getPartition()
        community_map = communities.subsetSizeMap()
        for community, size in community_map.items():
            LOGGER.debug("Got community with size: %s", size)
            community_members = list(
                communities.getMembers(community)
            )  # Need to do this to do batching.
            community_members = [component_node_map[i] for i in community_members]
            if strictness == "community":
                assignments.extend(
                    [
                        {"id": node_to_id_map[n], "cluster": cluster_index}
                        for n in community_members
                    ]
                )
                cluster_index += 1
                continue

            for start in range(0, len(community_members), max_clique_batch_size):
                community_nodes = community_members[
                    start : start + max_clique_batch_size
                ]
                LOGGER.debug("Creating subgraph with %s nodes.", len(community_nodes))
                # Map between node values for a community
                community_node_map = dict(enumerate(community_nodes))
                subgraph = nk.graphtools.subgraphFromNodes(
                    graph, community_nodes, compact=True
                )

                while subgraph.numberOfNodes() > 0:
                    LOGGER.debug("Subgraph size: %s", subgraph.numberOfNodes())
                    clique = nk.clique.MaximalCliques(subgraph, maximumOnly=True)
                    clique.run()
                    clique_members = clique.getCliques()[0]
                    assignments.extend(
                        [
                            {
                                "id": node_to_id_map[community_node_map[n]],
                                "cluster": cluster_index,
                            }
                            for n in clique_members
                        ]
                    )
                    cluster_index += 1
                    for n in clique_members:
                        subgraph.removeNode(n)

    return assignments
