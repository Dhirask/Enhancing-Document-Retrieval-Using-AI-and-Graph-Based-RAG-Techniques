import logging
import re
from typing import Dict, Iterable, List, Tuple

from .ingestion import Chunk, Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j-backed graph store with entity linking and typed traversal.

    Schema:
    - (:Chunk {chunk_id, text, source_document})
    - (:Entity {entity_id, name, type})
    - (:Document {doc_id})
    - (Entity)-[:mentions]->(Chunk)        — entity provenance
    - (Chunk)-[:part_of]->(Document)       — structural
    - (Entity)-[:co_occurs]->(Entity)      — co-occurrence (sentence-scoped)
    - (Entity)-[:svo_<verb>]->(Entity)     — subject-verb-object triples (e.g. svo_founded)
    """

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
        except ImportError as exc:
            raise ImportError("neo4j Python driver is required. Install with `pip install neo4j`.") from exc

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info(f"Connected to Neo4j at {self.uri}, database: {self.database}")
        self._verify_database()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j connection")
        self._driver = None

    def _verify_database(self) -> None:
        """Verify database exists and log active database."""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test").single()
                if result:
                    logger.info(f"Database '{self.database}' verified")
        except Exception as e:
            logger.warning(f"Could not verify database: {e}")

    def verify_connection(self) -> bool:
        """Test connectivity to Neo4j and log status."""
        if not self._driver:
            self.connect()
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test").single()
                logger.info(f"Neo4j connection verified on database '{self.database}'")
                return result is not None
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def clear_database(self) -> None:
        """Delete all nodes and relationships in the configured database (demo use)."""
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("GraphStore driver not initialized")
        try:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info(f"Cleared all data in Neo4j database '{self.database}'")
        except Exception as exc:
            logger.error(f"Failed to clear Neo4j database '{self.database}': {exc}")
            raise

    # ------------------------------------------------------------------ #
    #  Write Operations                                                    #
    # ------------------------------------------------------------------ #
    def upsert(self, chunks: Iterable[Chunk], entities: Iterable[Entity], relations: Iterable[Relation]) -> None:
        """Upsert chunks, entities, and relations into Neo4j."""
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("Failed to initialize Neo4j driver")

        chunks_list = list(chunks)
        entities_list = list(entities)
        relations_list = list(relations)

        logger.info(f"Upserting {len(chunks_list)} chunks, {len(entities_list)} entities, {len(relations_list)} relations")

        if not chunks_list and not entities_list and not relations_list:
            logger.warning("No data to upsert")
            return

        with self._driver.session(database=self.database) as session:
            if chunks_list:
                session.execute_write(self._upsert_chunks, chunks_list)
                logger.debug(f"Upserted {len(chunks_list)} chunks")
            if entities_list:
                session.execute_write(self._upsert_entities, entities_list)
                logger.debug(f"Upserted {len(entities_list)} entities")
            if relations_list:
                session.execute_write(self._upsert_relations, relations_list)
                logger.debug(f"Upserted {len(relations_list)} relations")

    @staticmethod
    def _upsert_chunks(tx, chunks: List[Chunk]) -> None:
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {chunk_id: row.id}) "
            "SET c.text = row.text, c.source_document = row.source_document"
        )
        tx.run(query, rows=[chunk.__dict__ for chunk in chunks])

    @staticmethod
    def _upsert_entities(tx, entities: List[Entity]) -> None:
        query = (
            "UNWIND $rows AS row "
            "MERGE (e:Entity {entity_id: row.id}) "
            "SET e.name = row.label, e.type = row.type"
        )
        tx.run(query, rows=[entity.__dict__ for entity in entities])

    @staticmethod
    def _upsert_relations(tx, relations: List[Relation]) -> None:
        """Upsert relations using label-aware MERGE to avoid node ambiguity.

        Handles dynamic svo_<verb> types by validating the identifier pattern.
        """
        rels_by_type = {}
        for rel in relations:
            rels_by_type.setdefault(rel.type, []).append(rel)

        for rel_type, rel_list in rels_by_type.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rel_type):
                raise ValueError(f"Invalid relation type: {rel_type}")

            if rel_type == "part_of":
                # Chunk -> Document
                rel_query = (
                    "UNWIND $rows AS row "
                    "MATCH (h:Chunk {chunk_id: row.head}) "
                    "MERGE (t:Document {doc_id: row.tail}) "
                    f"MERGE (h)-[r:`{rel_type}`]->(t) "
                    "SET r.score = row.score"
                )
            elif rel_type == "mentions":
                # Entity -> Chunk
                rel_query = (
                    "UNWIND $rows AS row "
                    "MATCH (h:Entity {entity_id: row.head}) "
                    "MATCH (t:Chunk {chunk_id: row.tail}) "
                    f"MERGE (h)-[r:`{rel_type}`]->(t) "
                    "SET r.score = row.score"
                )
            else:
                # Entity -> Entity (co_occurs, svo_founded, svo_acquired, etc.)
                rel_query = (
                    "UNWIND $rows AS row "
                    "MATCH (h:Entity {entity_id: row.head}) "
                    "MATCH (t:Entity {entity_id: row.tail}) "
                    f"MERGE (h)-[r:`{rel_type}`]->(t) "
                    "SET r.score = row.score"
                )
            tx.run(rel_query, rows=[rel.__dict__ for rel in rel_list])

    # ------------------------------------------------------------------ #
    #  Entity Linking: surface form → chunk IDs                            #
    # ------------------------------------------------------------------ #
    def find_entry_chunks(self, entity_names: List[str], limit: int = 20) -> List[str]:
        """Resolve entity surface forms to chunk IDs via the graph.

        Matches entity names (case-insensitive, contains) and follows MENTIONS edges
        to find the chunks those entities appear in.
        """
        if not entity_names:
            return []
        if not self._driver:
            self.connect()
        if self._driver is None:
            return []

        with self._driver.session(database=self.database) as session:
            result = session.execute_read(self._find_entry_chunks_query, entity_names, limit)
            logger.info(f"Entity linking: {entity_names} -> {len(result)} entry chunks")
            return result

    @staticmethod
    def _find_entry_chunks_query(tx, entity_names: List[str], limit: int) -> List[str]:
        """Find chunks associated with entities whose names match the given surface forms."""
        query = (
            "UNWIND $names AS name "
            "MATCH (e:Entity)-[:mentions]->(c:Chunk) "
            "WHERE toLower(e.name) CONTAINS toLower(name) "
            "RETURN DISTINCT c.chunk_id AS chunk_id LIMIT $limit"
        )
        records = tx.run(query, names=entity_names, limit=limit)
        return [rec["chunk_id"] for rec in records if rec.get("chunk_id")]

    # ------------------------------------------------------------------ #
    #  Graph Traversal with Relation Type Filtering                        #
    # ------------------------------------------------------------------ #
    def neighbors(self, node_ids: List[str], max_hops: int = 2, limit: int = 20,
                  relation_types: List[str] = None) -> List[Tuple[str, int]]:
        """Find neighboring chunk IDs via typed graph traversal.

        Returns list of (chunk_id, hop_distance) tuples.
        """
        if not node_ids:
            return []
        if not self._driver:
            self.connect()
        if self._driver is None:
            return []

        with self._driver.session(database=self.database) as session:
            result = session.execute_read(
                self._neighbors_query, node_ids, max_hops, limit, relation_types or []
            )
            return result

    @staticmethod
    def _neighbors_query(tx, node_ids: List[str], max_hops: int, limit: int,
                         relation_types: List[str]) -> List[Tuple[str, int]]:
        """Multi-hop traversal from entry chunks through entity-entity edges.

        Uses WHERE clause to filter relation types dynamically, supporting
        both exact types (co_occurs) and prefix patterns (svo_*).
        When relation_types is empty, defaults to co_occurs + svo_* prefix.
        Returns (chunk_id, actual_hop_distance) pairs.
        """
        # Split relation_types into exact matches and prefix patterns (ending with *)
        exact_types: List[str] = []
        prefixes: List[str] = []

        if relation_types:
            for rt in relation_types:
                safe = re.sub(r"[^A-Za-z0-9_*]", "", rt)
                if safe.endswith("*"):
                    prefixes.append(safe[:-1])  # strip the trailing *
                else:
                    exact_types.append(safe)
        else:
            # Default: co_occurs (exact) + svo_ (prefix)
            exact_types = ["co_occurs"]
            prefixes = ["svo_"]

        # Build the WHERE predicate for relationship filtering
        type_conditions = []
        if exact_types:
            type_conditions.append("type(r) IN $exactTypes")
        if prefixes:
            type_conditions.append(
                "any(pfx IN $prefixes WHERE type(r) STARTS WITH pfx)"
            )
        type_filter = " OR ".join(type_conditions) if type_conditions else "true"

        query = (
            "MATCH (start:Chunk) WHERE start.chunk_id IN $node_ids "
            "MATCH (start)<-[:mentions]-(e1:Entity) "
            f"MATCH path = (e1)-[rels*1..{max_hops}]-(e2:Entity) "
            f"WHERE ALL(r IN rels WHERE {type_filter}) "
            "WITH e2, length(path) AS hops "
            "MATCH (e2)-[:mentions]->(target:Chunk) "
            "WHERE NOT target.chunk_id IN $node_ids "
            "RETURN DISTINCT target.chunk_id AS chunk_id, min(hops) AS hops "
            "LIMIT $limit"
        )

        records = tx.run(
            query,
            node_ids=node_ids,
            limit=limit,
            exactTypes=exact_types,
            prefixes=prefixes,
        )
        return [(rec["chunk_id"], rec["hops"]) for rec in records if rec.get("chunk_id")]

    # ------------------------------------------------------------------ #
    #  Subgraph Retrieval for Generation Context                           #
    # ------------------------------------------------------------------ #
    def get_subgraph_triples(self, chunk_ids: List[str], limit: int = 30) -> List[Dict]:
        """Retrieve knowledge graph triples connected to the given chunks.

        Returns dicts: {subject, predicate, object, subject_type, object_type}
        """
        if not chunk_ids:
            return []
        if not self._driver:
            self.connect()
        if self._driver is None:
            return []

        with self._driver.session(database=self.database) as session:
            return session.execute_read(self._subgraph_triples_query, chunk_ids, limit)

    @staticmethod
    def _subgraph_triples_query(tx, chunk_ids: List[str], limit: int) -> List[Dict]:
        """Get entity-entity relations from chunks for graph-aware generation.

        Matches both co_occurs and any svo_<verb> relations.
        Returns triples with human-readable verb extracted from type name.
        """
        query = (
            "MATCH (e1:Entity)-[:mentions]->(c:Chunk) "
            "WHERE c.chunk_id IN $chunk_ids "
            "MATCH (e1)-[r]->(e2:Entity) "
            "WHERE type(r) = 'co_occurs' OR type(r) STARTS WITH 'svo_' "
            "RETURN DISTINCT e1.name AS subject, type(r) AS predicate, e2.name AS object, "
            "  e1.type AS subject_type, e2.type AS object_type, c.chunk_id AS chunk_id "
            "LIMIT $limit"
        )
        records = tx.run(query, chunk_ids=chunk_ids, limit=limit)
        results = []
        for rec in records:
            triple = dict(rec)
            # Make predicate human-readable: svo_founded -> "founded", co_occurs -> "related to"
            raw_pred = triple.get("predicate", "")
            if raw_pred.startswith("svo_"):
                triple["predicate_label"] = raw_pred[4:].replace("_", " ")
            elif raw_pred == "co_occurs":
                triple["predicate_label"] = "related to"
            else:
                triple["predicate_label"] = raw_pred
            results.append(triple)
        return results

    # ------------------------------------------------------------------ #
    #  Centrality (Normalized Degree)                                      #
    # ------------------------------------------------------------------ #
    def centrality_score(self, node_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate normalized degree centrality for chunks."""
        if not node_ids:
            return []
        if not self._driver:
            self.connect()
        if self._driver is None:
            return []

        with self._driver.session(database=self.database) as session:
            records = session.execute_read(self._degree_query, node_ids)
            return records

    @staticmethod
    def _degree_query(tx, node_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate normalized degree centrality.

        Normalizes by dividing each node's degree by the max degree in the result set,
        producing values in [0, 1].
        """
        query = (
            "MATCH (n:Chunk) "
            "WHERE n.chunk_id IN $node_ids "
            "WITH n, size([(n)--() | 1]) AS degree "
            "WITH collect({chunk_id: n.chunk_id, degree: degree}) AS nodes, "
            "     max(degree) AS max_degree "
            "UNWIND nodes AS node "
            "RETURN node.chunk_id AS chunk_id, "
            "  CASE WHEN max_degree > 0 THEN toFloat(node.degree) / max_degree ELSE 0.0 END AS degree"
        )
        records = tx.run(query, node_ids=node_ids)
        results = [(rec["chunk_id"], float(rec["degree"])) for rec in records if rec.get("chunk_id")]
        logger.debug(f"Normalized centrality scores for {len(results)} nodes")
        return results
