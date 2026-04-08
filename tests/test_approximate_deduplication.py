import perception.approximate_deduplication as ad


def get_cluster_members(assignments):
    clusters: dict[int, list[str]] = {}
    for assignment in assignments:
        clusters.setdefault(assignment["cluster"], []).append(assignment["id"])
    return sorted(sorted(members) for members in clusters.values())


def test_pairs_to_clusters_component_strictness():
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c", "d"],
        pairs=[("a", "b"), ("b", "c")],
        strictness="component",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"], ["d"]]


def test_pairs_to_clusters_community_strictness():
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c"],
        pairs=[("a", "b"), ("b", "c")],
        strictness="community",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"]]


def test_pairs_to_clusters_clique_strictness():
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c", "d"],
        pairs=[("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")],
        strictness="clique",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"], ["d"]]
