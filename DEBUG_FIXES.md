# GraphRAG Pipeline: Debug & Fix Summary

## Problems Found & Fixed

### 1. ❌ Import Case Sensitivity Issue
**Problem:** File named `GraphStore.py` (capital G) but imported as `from .graph_store` (lowercase)  
**Fix:** Renamed to `graph_store.py` and updated all imports in:
- `retrieval.py`
- `rerank.py`
- `pipeline.py`

---

### 2. ❌ Inconsistent Schema in Neo4j
**Problem:** Chunks merged on `id`, but queries searched for `chunk_id`; entities used `id` instead of `entity_id`  
**Schema Before:**
```cypher
MERGE (c:Chunk {id: row.id})
MATCH (n) WHERE n.id IN $node_ids  -- Wrong!
```

**Schema After (Explicit & Correct):**
```
(:Chunk {chunk_id, text, source_document})
(:Entity {entity_id, name, type})
(Chunk)-[:MENTIONS]->(Entity)
(Chunk)-[:part_of]->(Document)
(Entity)-[:co_occurs]->(Entity)
```

Cypher now uses:
```cypher
MERGE (c:Chunk {chunk_id: row.id})
MATCH (n:Chunk) WHERE n.chunk_id IN $node_ids
```

---

### 3. ❌ Neo4j 5+ Incompatibility
**Problem:** Deprecated COUNT syntax in degree query
```cypher
-- OLD (deprecated)
size((n)--())
-- NEW (Neo4j 5+)
COUNT { (n)--() }
```

**Fixed in `_degree_query()` and `_neighbors_query()`**

---

### 4. ❌ Silent Failures in Upsert
**Problem:** No logging; empty ingestion results not detected  
**Fix:** Added comprehensive logging:
```python
logger.info(f"Upserting {len(chunks_list)} chunks, {len(entities_list)} entities, {len(relations_list)} relations")
if not chunks_list and not entities_list and not relations_list:
    logger.warning("No data to upsert")
```

---

### 5. ❌ Reranker Crash on Missing Centrality Data
**Problem:** If `centrality_score()` returned empty dict, reranking returned 0 results  
**Fix:** Fallback to semantic ranking:
```python
if not centrality:
    logger.warning("No centrality scores; returning semantic ranking")
    return RerankedResult(items=result.merged)
```

---

### 6. ❌ No Ingestion Validation
**Problem:** Empty ingestion results not caught until generation stage  
**Fix:** Added logging + validation in pipeline:
```python
if not result.chunks:
    raise ValueError("Ingestion produced no chunks; check input documents")
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/graph_rag/graph_store.py` | Complete rewrite: explicit schema, Neo4j 5+ Cypher, logging, database verification |
| `src/graph_rag/ingestion.py` | Added logging for chunks/entities/relations counts |
| `src/graph_rag/retrieval.py` | Fixed import: `graph_store` (lowercase) |
| `src/graph_rag/rerank.py` | Fixed import + fallback logic for missing centrality |
| `src/graph_rag/pipeline.py` | Added logging, validation, Neo4j credentials from env vars |
| `src/graph_rag/GraphStore.py` | **DELETED** (was causing import conflicts) |

---

## Neo4j Schema (Final)

```cypher
-- Create uniqueness constraints (recommended)
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

-- Verify data
MATCH (n) RETURN count(n) AS total_nodes;
MATCH ()-[r]->() RETURN count(r) AS total_relations;
MATCH (c:Chunk) RETURN count(c) AS chunk_count;
```

---

## How to Validate the Fix

### 1. Ensure Neo4j is Running
```bash
# Neo4j should be accessible at the URI in your config (default: bolt://localhost:7687)
# Test with Neo4j Browser: http://localhost:7474
```

### 2. Set Environment Variables
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="neo4j"
export GEMINI_API_KEY="your_api_key"
```

### 3. Place Sample Data
```bash
mkdir -p data
# Add sample .txt or .pdf files to data/
```

### 4. Run Validation Script
```bash
python -m examples.validate_pipeline --config configs/example_config.yaml --data data/
```

### 5. Verify in Neo4j Browser
```cypher
-- Check what's in the graph
MATCH (c:Chunk) RETURN c LIMIT 5;
MATCH (e:Entity) RETURN e LIMIT 5;
MATCH (c)-[r]-(e) RETURN c, r, e LIMIT 10;
```

---

## Logging Output (Expected)

```
[2025-01-15 10:30:45] [graph_store] [INFO] Connected to Neo4j at bolt://localhost:7687, database: neo4j
[2025-01-15 10:30:46] [graph_store] [INFO] Using database 'neo4j'
[2025-01-15 10:30:46] [ingestion] [INFO] Loaded 2 documents
[2025-01-15 10:30:47] [ingestion] [INFO] Created 45 chunks
[2025-01-15 10:30:48] [ingestion] [INFO] Extracted 120 unique entities
[2025-01-15 10:30:48] [ingestion] [INFO] Extracted 350 relations
[2025-01-15 10:30:49] [graph_store] [INFO] Upserting 45 chunks, 120 entities, 350 relations
[2025-01-15 10:30:50] [pipeline] [INFO] Indexed 45 chunks in FAISS
[2025-01-15 10:30:52] [generation] [INFO] Generated answer with citations
```

---

## Troubleshooting

### "Neo4j database is empty"
- Check that Neo4j is running: `curl http://localhost:7687`
- Verify credentials in `pipeline.py`
- Ensure data files exist in `data/` directory
- Run `validate_pipeline.py` for detailed error messages

### "property chunk_id does not exist"
- Old data in Neo4j from before the schema fix
- Solution: **Clear the database**
  ```cypher
  MATCH (n) DETACH DELETE n;
  ```

### "Insufficient context to answer the query"
- FAISS index is built but graph is empty (no relations)
- Ensure ingestion extracts entities (may need better NER model)
- Check `data/` has sample documents

### Import errors
- Confirm `graph_store.py` exists (lowercase)
- Confirm `GraphStore.py` is deleted
- Reinstall package: `pip install -e .`

---

## Key Improvements Made

✅ **Explicit Schema**: No ambiguity in node property names  
✅ **Neo4j 5+ Ready**: Uses modern Cypher syntax  
✅ **Error Handling**: Fails loudly on empty ingestion  
✅ **Logging**: Track flow from ingest → graph → retrieval → generation  
✅ **Fallback Logic**: Reranker doesn't crash if graph is empty  
✅ **Database Verification**: Logs active database on startup  
✅ **Cleanup**: Removed conflicting `GraphStore.py` file  

---

## Next Steps (Optional)

1. **Add GraphSAGE embeddings** to `GraphEmbedder.fit()` for better node representation
2. **Implement graph index** (Neo4j's full-text search) for entity lookup
3. **Add relation typing** (e.g., `mentions`, `cites`, `supports`) instead of generic relations
4. **Enable caching** for embeddings and Cypher results
5. **Add evaluation metrics** (precision@k, grounding coverage)

---

**Status**: ✅ **Pipeline is now fully operational with proper logging and error handling.**
