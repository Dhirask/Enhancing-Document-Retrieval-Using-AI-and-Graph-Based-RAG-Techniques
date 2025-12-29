from typing import Iterable, List, Tuple

from .ingestion import Chunk, Entity, Relation


class GraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None  # lazy init placeholder

    def connect(self) -> None:
        # Placeholder for Neo4j driver init to avoid dependency here.
        self._driver = True

    def close(self) -> None:
        self._driver = None

    def upsert(self, chunks: Iterable[Chunk], entities: Iterable[Entity], relations: Iterable[Relation]) -> None:
        # In a real implementation, run Cypher MERGE statements in batches.
        if not self._driver:
            self.connect()
        # Here we assume the graph DB is updated.

    def neighbors(self, node_ids: List[str], max_hops: int = 2, limit: int = 20) -> List[str]:
        # Placeholder traversal; a real traversal would use variable-length patterns.
        seen = []
        for node_id in node_ids:
            seen.append(node_id)
        return seen[:limit]

    def centrality_score(self, node_ids: List[str]) -> List[Tuple[str, float]]:
        # Placeholder centrality; a real system would run graph algo procedures.
        return [(node_id, 1.0) for node_id in node_ids]
