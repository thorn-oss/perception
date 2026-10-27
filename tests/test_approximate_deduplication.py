from types import SimpleNamespace

import pytest

import perception.approximate_deduplication as ad
from perception.approximate_deduplication import _graph_backend


@pytest.fixture(params=["default", "networkx"])
def graph_backend(request, monkeypatch):
    if request.param == "networkx":
        monkeypatch.setattr(
            ad, "get_graph_backend", _graph_backend.NetworkxGraphBackend
        )


@pytest.mark.parametrize("python_minor", range(11, 16))
@pytest.mark.parametrize(
    "sys_platform,machine,expected_backend",
    [
        ("linux", "x86_64", "networkit"),
        ("linux", "aarch64", "networkit"),
        ("linux", "arm64", "networkx"),
        ("linux", "armv7l", "networkx"),
        ("win32", "AMD64", "networkit"),
        ("win32", "ARM64", "networkx"),
        ("darwin", "x86_64", "networkx"),
        ("darwin", "arm64", "networkx"),
    ],
)
def test_graph_backend_selection(
    monkeypatch, sys_platform, machine, expected_backend, python_minor
):
    monkeypatch.setattr(
        _graph_backend,
        "sys",
        SimpleNamespace(platform=sys_platform, version_info=(3, python_minor)),
    )
    monkeypatch.setattr(
        _graph_backend, "platform", SimpleNamespace(machine=lambda: machine)
    )
    monkeypatch.setattr(_graph_backend, "NetworkitGraphBackend", lambda: "networkit")
    monkeypatch.setattr(_graph_backend, "NetworkxGraphBackend", lambda: "networkx")

    if sys_platform == "win32" and python_minor == 15:
        expected_backend = "networkx"
    assert _graph_backend.get_graph_backend() == expected_backend


def get_cluster_members(assignments):
    clusters: dict[int, list[str]] = {}
    for assignment in assignments:
        clusters.setdefault(assignment["cluster"], []).append(assignment["id"])
    return sorted(sorted(members) for members in clusters.values())


def test_pairs_to_clusters_component_strictness(graph_backend):
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c", "d"],
        pairs=[("a", "b"), ("b", "c")],
        strictness="component",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"], ["d"]]


def test_pairs_to_clusters_community_strictness(graph_backend):
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c"],
        pairs=[("a", "b"), ("b", "c")],
        strictness="community",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"]]


def test_pairs_to_clusters_clique_strictness(graph_backend):
    assignments = ad.pairs_to_clusters(
        ids=["a", "b", "c", "d"],
        pairs=[("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")],
        strictness="clique",
    )

    assert get_cluster_members(assignments) == [["a", "b", "c"], ["d"]]
