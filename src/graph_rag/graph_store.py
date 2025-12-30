import logging
import re
from typing import Iterable, List, Tuple

from .ingestion import Chunk, Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j-backed graph store with explicit schema and Neo4j 5+ compatibility.
    
    Schema:
    - (:Chunk {chunk_id, text, source_document})
    - (:Entity {entity_id, name, type})
    - (Chunk)-[:MENTIONS]->(Entity)
    - (Chunk)-[:part_of]->(Document)
    - (Entity)-[:co_occurs]->(Entity)
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
            with self._driver.session(database="neo4j") as session:
                dbs = session.run("SHOW DATABASES").data()
                db_names = [db["name"] for db in dbs]
                if self.database in db_names:
                    logger.info(f"Using database '{self.database}'")
                else:
                    logger.warning(f"Database '{self.database}' not found. Available: {db_names}")
        except Exception as e:
            logger.warning(f"Could not verify database: {e}")

    # --- Write ---
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
        """Insert or update chunks with explicit chunk_id property."""
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {chunk_id: row.id}) "
            "SET c.text = row.text, c.source_document = row.source_document"
        )
        tx.run(query, rows=[chunk.__dict__ for chunk in chunks])

    @staticmethod
    def _upsert_entities(tx, entities: List[Entity]) -> None:
        """Insert or update entities with explicit entity_id property."""
        query = (
            "UNWIND $rows AS row "
            "MERGE (e:Entity {entity_id: row.id}) "
            "SET e.name = row.label, e.type = row.type"
        )
        tx.run(query, rows=[entity.__dict__ for entity in entities])

    @staticmethod
    def _upsert_relations(tx, relations: List[Relation]) -> None:
        """Upsert relations, grouping by type for safe Cypher construction."""
        rels_by_type = {}
        for rel in relations:
            rels_by_type.setdefault(rel.type, []).append(rel)
        
        for rel_type, rel_list in rels_by_type.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rel_type):
                raise ValueError(f"Invalid relation type: {rel_type}")

            rel_query = (
                "UNWIND $rows AS row "
                "MERGE (h {id: row.head}) "
                "MERGE (t {id: row.tail}) "
                f"MERGE (h)-[r:`{rel_type}`]->(t) "
                "SET r.score = row.score"
            )
            tx.run(rel_query, rows=[rel.__dict__ for rel in rel_list])

    # --- Read ---
    def neighbors(self, node_ids: List[str], max_hops: int = 2, limit: int = 20) -> List[str]:
        """Find neighboring chunk IDs via graph traversal."""
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("GraphStore driver not initialized")

        with self._driver.session(database=self.database) as session:
            result = session.execute_read(self._neighbors_query, node_ids, max_hops, limit)
            return result

    @staticmethod
    def _neighbors_query(tx, node_ids: List[str], max_hops: int, limit: int) -> List[str]:
        """Find neighbors via variable-length pattern (Neo4j 5+ compatible)."""
        query = (
            "MATCH (n:Chunk) WHERE n.chunk_id IN $node_ids "
            f"MATCH (n)-[*1..{max_hops}]-(m:Chunk) "
            "RETURN DISTINCT m.chunk_id AS chunk_id LIMIT $limit"
        )
        records = tx.run(query, node_ids=node_ids, limit=limit)
        return [rec["chunk_id"] for rec in records if rec.get("chunk_id")]

    def centrality_score(self, node_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate degree centrality for chunks."""
        if not self._driver:
            self.connect()
        if self._driver is None:
            raise RuntimeError("GraphStore driver not initialized")

        with self._driver.session(database=self.database) as session:
            records = session.execute_read(self._degree_query, node_ids)
            return records

    @staticmethod
    def _degree_query(tx, node_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate node degree (centrality) via Neo4j 5+ COUNT {{ }} syntax."""
        query = (
            "MATCH (n:Chunk) "
            "WHERE n.chunk_id IN $node_ids "
            "RETURN n.chunk_id AS chunk_id, COUNT { (n)--() } AS degree"
        )
        records = tx.run(query, node_ids=node_ids)
        results = [(rec["chunk_id"], float(rec["degree"])) for rec in records if rec.get("chunk_id")]
        logger.debug(f"Centrality scores for {len(results)} nodes")
        return results
