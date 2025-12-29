import re
from typing import Iterable, List, Tuple

from .ingestion import Chunk, Entity, Relation


class GraphStore:
    """Neo4j-backed graph store with MERGE-based upserts and simple traversals."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    def connect(self) -> None:
        if self._driver:
            return
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("neo4j Python driver is required. Install with `pip install neo4j`.") from exc
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        if self._driver:
            self._driver.close()
        self._driver = None

    # --- Write ---
    def upsert(self, chunks: Iterable[Chunk], entities: Iterable[Entity], relations: Iterable[Relation]) -> None:
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("Failed to initialize Neo4j driver")

        with self._driver.session(database=self.database) as session:
            session.execute_write(self._upsert_chunks, list(chunks))
            session.execute_write(self._upsert_entities, list(entities))
            session.execute_write(self._upsert_relations, list(relations))

    @staticmethod
    def _upsert_chunks(tx, chunks: List[Chunk]) -> None:
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {id: row.id}) "
            "SET c.text = row.text, c.source_document = row.source_document"
        )
        tx.run(query, rows=[chunk.__dict__ for chunk in chunks])

    @staticmethod
    def _upsert_entities(tx, entities: List[Entity]) -> None:
        query = (
            "UNWIND $rows AS row "
            "MERGE (e:Entity {id: row.id}) "
            "SET e.label = row.label, e.type = row.type"
        )
        tx.run(query, rows=[entity.__dict__ for entity in entities])

    @staticmethod
    def _upsert_relations(tx, relations: List[Relation]) -> None:
        # Neo4j does not allow parameterizing the relationship type directly in Cypher text.
        # We therefore dispatch per distinct relation type.
        rels_by_type = {}
        for rel in relations:
            rels_by_type.setdefault(rel.type, []).append(rel)
        for rel_type, rel_list in rels_by_type.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rel_type):
                raise ValueError(f"Invalid relation type: {rel_type}")

            if rel_type == "part_of":
                head_label, tail_label = "Chunk", "Document"
            elif rel_type == "mentions":
                head_label, tail_label = "Entity", "Chunk"
            elif rel_type == "co_occurs":
                head_label, tail_label = "Entity", "Entity"
            else:
                head_label, tail_label = "Entity", "Entity"

            rel_query = (
                "UNWIND $rows AS row "
                f"MERGE (h:{head_label} {{id: row.head}}) "
                f"MERGE (t:{tail_label} {{id: row.tail}}) "
                f"MERGE (h)-[r:`{rel_type}`]->(t) "
                "SET r.score = row.score"
            )
            tx.run(rel_query, rows=[rel.__dict__ for rel in rel_list])

    # --- Read ---
    def neighbors(self, node_ids: List[str], max_hops: int = 2, limit: int = 20) -> List[str]:
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("GraphStore driver not initialized")

        with self._driver.session(database=self.database) as session:
            result = session.execute_read(self._neighbors_query, node_ids, max_hops, limit)
            return result

    @staticmethod
    def _neighbors_query(tx, node_ids: List[str], max_hops: int, limit: int) -> List[str]:
        query = (
            "MATCH (n) WHERE n.id IN $node_ids "
            f"MATCH (n)-[*1..{max_hops}]-(m) "

            "RETURN DISTINCT m.id AS id LIMIT $limit"
        )
        records = tx.run(query, node_ids=node_ids, limit=limit)
        return [rec["id"] for rec in records]

    def centrality_score(self, node_ids: List[str]) -> List[Tuple[str, float]]:
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("GraphStore driver not initialized")

        with self._driver.session(database=self.database) as session:
            records = session.execute_read(self._degree_query, node_ids)
            return records

    @staticmethod
    def _degree_query(tx, node_ids: List[str]) -> List[Tuple[str, float]]:
        query = (
            "MATCH (n) WHERE n.id IN $node_ids "
            "RETURN n.id AS id, size((n)--()) AS degree"
        )
        records = tx.run(query, node_ids=node_ids)
        return [(rec["id"], float(rec["degree"])) for rec in records]
