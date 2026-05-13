import sys
import typing
from abc import ABC, abstractmethod


class GraphBackend(ABC):
    @abstractmethod
    def build_graph(
        self, node_count: int, edges: typing.Iterable[tuple[int, int]]
    ) -> typing.Any: ...

    @abstractmethod
    def connected_components(self, graph: typing.Any) -> list[list[int]]: ...

    @abstractmethod
    def communities(
        self, graph: typing.Any, component: list[int]
    ) -> list[list[int]]: ...

    @abstractmethod
    def maximal_cliques(
        self,
        graph: typing.Any,
        community_nodes: list[int],
        max_clique_batch_size: int,
    ) -> list[list[int]]: ...


_EXTRA_INSTALL_HINT = (
    "perception.approximate_deduplication requires the "
    "'approximate-deduplication' extra. Install it with "
    "`pip install perception[approximate-deduplication]`."
)


class NetworkitGraphBackend(GraphBackend):
    def __init__(self):
        try:
            import networkit as nk
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without extra installed
            raise ImportError(_EXTRA_INSTALL_HINT) from exc

        self.nk = nk

    def build_graph(
        self, node_count: int, edges: typing.Iterable[tuple[int, int]]
    ) -> typing.Any:
        graph = self.nk.Graph(node_count)
        for start, end in edges:
            graph.addEdge(start, end)
        return graph

    def connected_components(self, graph: typing.Any) -> list[list[int]]:
        cc_query = self.nk.components.ConnectedComponents(graph)
        cc_query.run()
        return cc_query.getComponents()

    def communities(self, graph: typing.Any, component: list[int]) -> list[list[int]]:
        component_node_map = dict(enumerate(component))
        subgraph = self.nk.graphtools.subgraphFromNodes(graph, component, compact=True)
        algo = self.nk.community.PLP(subgraph, maxIterations=32)
        algo.run()
        communities = algo.getPartition()
        return [
            [component_node_map[node] for node in communities.getMembers(community)]
            for community in communities.subsetSizeMap().keys()
        ]

    def maximal_cliques(
        self,
        graph: typing.Any,
        community_nodes: list[int],
        max_clique_batch_size: int,
    ) -> list[list[int]]:
        cliques: list[list[int]] = []
        for start in range(0, len(community_nodes), max_clique_batch_size):
            batch_nodes = community_nodes[start : start + max_clique_batch_size]
            community_node_map = dict(enumerate(batch_nodes))
            subgraph = self.nk.graphtools.subgraphFromNodes(
                graph, batch_nodes, compact=True
            )

            while subgraph.numberOfNodes() > 0:
                clique = self.nk.clique.MaximalCliques(subgraph, maximumOnly=True)
                clique.run()
                clique_members = clique.getCliques()[0]
                cliques.append([community_node_map[node] for node in clique_members])
                for node in clique_members:
                    subgraph.removeNode(node)

        return cliques


class NetworkxGraphBackend(GraphBackend):
    def __init__(self):
        try:
            import networkx as nx
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without extra installed
            raise ImportError(_EXTRA_INSTALL_HINT) from exc

        self.nx = nx

    def build_graph(
        self, node_count: int, edges: typing.Iterable[tuple[int, int]]
    ) -> typing.Any:
        graph = self.nx.Graph()
        graph.add_nodes_from(range(node_count))
        graph.add_edges_from(edges)
        return graph

    def connected_components(self, graph: typing.Any) -> list[list[int]]:
        return [list(component) for component in self.nx.connected_components(graph)]

    def communities(self, graph: typing.Any, component: list[int]) -> list[list[int]]:
        subgraph = graph.subgraph(component)
        return [
            list(community)
            for community in self.nx.algorithms.community.asyn_lpa_communities(
                subgraph, seed=0
            )
        ]

    def maximal_cliques(
        self,
        graph: typing.Any,
        community_nodes: list[int],
        max_clique_batch_size: int,
    ) -> list[list[int]]:
        cliques: list[list[int]] = []
        for start in range(0, len(community_nodes), max_clique_batch_size):
            batch_nodes = community_nodes[start : start + max_clique_batch_size]
            subgraph = graph.subgraph(batch_nodes).copy()

            while subgraph.number_of_nodes() > 0:
                clique_members = max(
                    self.nx.find_cliques(subgraph),
                    key=lambda clique: (
                        len(clique),
                        tuple(sorted(clique)),
                    ),
                )
                cliques.append(list(clique_members))
                subgraph.remove_nodes_from(clique_members)

        return cliques


def get_graph_backend() -> GraphBackend:
    if sys.platform == "darwin":
        return NetworkxGraphBackend()
    return NetworkitGraphBackend()
