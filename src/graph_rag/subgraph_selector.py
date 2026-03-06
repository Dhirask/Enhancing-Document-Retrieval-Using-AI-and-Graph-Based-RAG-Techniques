"""
Knowledge-graph reasoning subgraph selector.

Given a user query and a list of raw triples from Neo4j, selects a small
connected subgraph (5-10 triples) that forms multi-hop reasoning paths
relevant to the query.

Algorithm (5 steps):
    1. Identify query entities / concepts
    2. Find seed nodes — triple endpoints matching query entities
    3. Graph traversal — BFS outward from seeds, scoring by relevance
    4. Construct reasoning subgraph — pick top connected triples
    5. Return the selected triples
"""

import logging
import re
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Maximum triples in the reasoning subgraph
MAX_SUBGRAPH_TRIPLES = 10
MIN_SUBGRAPH_TRIPLES = 3

# Cached spaCy model (lazy-loaded once on first use)
_nlp = None


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _token_set(text: str) -> Set[str]:
    """Return the set of non-trivial tokens in normalised text."""
    STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was", "were"}
    return {w for w in _normalise(text).split() if w and w not in STOP}


def _predicate_quality(predicate: str) -> float:
    """Return a quality score for a relation predicate.

    SVO-typed predicates (svo_*) are most informative, co_occurs is moderate,
    and unknown predicates get a low default score.
    """
    if predicate.startswith("svo_"):
        return 1.5
    if predicate == "co_occurs":
        return 0.5
    # Unknown / structural predicates
    return 0.1


# ------------------------------------------------------------------ #
#  Step 1 — Extract query concepts                                     #
# ------------------------------------------------------------------ #

def _extract_query_concepts(query: str) -> List[str]:
    """Break the query into meaningful concept phrases.

    Uses spaCy if available (NER + noun chunks), otherwise falls back to
    simple token-level extraction.
    """
    try:
        global _nlp
        if _nlp is None:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        doc = _nlp(query)
        concepts: List[str] = []
        # Named entities first
        for ent in doc.ents:
            concepts.append(ent.text.strip())
        # Noun chunks as fallback
        STOP_CHUNKS = {"it", "this", "that", "what", "which", "who", "how", "the", "a", "an"}
        for nc in doc.noun_chunks:
            text = nc.text.strip()
            if text.lower() not in STOP_CHUNKS and len(text) > 1 and text not in concepts:
                concepts.append(text)
        if concepts:
            return concepts
    except Exception:
        pass

    # Fallback: split on whitespace, remove stop words
    return [w for w in _token_set(query) if len(w) > 2]


# ------------------------------------------------------------------ #
#  Step 2 — Find seed nodes                                            #
# ------------------------------------------------------------------ #

def _entity_matches_concept(entity: str, concept_tokens: Set[str]) -> float:
    """Score how well an entity matches a query concept (0-1).

    Uses token-overlap (Jaccard-like) for fuzzy matching.
    """
    ent_tokens = _token_set(entity)
    if not ent_tokens or not concept_tokens:
        return 0.0
    overlap = ent_tokens & concept_tokens
    if not overlap:
        return 0.0
    # Weighted: how much of the concept is covered
    return len(overlap) / max(len(concept_tokens), 1)


def _find_seed_entities(
    triples: List[Dict], query_concepts: List[str]
) -> Dict[str, float]:
    """Identify entities in the triples that match query concepts.

    Returns {entity_name: relevance_score}.
    """
    concept_token_sets = [_token_set(c) for c in query_concepts]

    # Collect all unique entities from triples
    all_entities: Set[str] = set()
    for t in triples:
        all_entities.add(t.get("subject", ""))
        all_entities.add(t.get("object", ""))
    all_entities.discard("")

    seed_scores: Dict[str, float] = {}
    for entity in all_entities:
        best_score = 0.0
        for ctokens in concept_token_sets:
            score = _entity_matches_concept(entity, ctokens)
            best_score = max(best_score, score)
        if best_score > 0.0:
            seed_scores[entity] = best_score

    return seed_scores


# ------------------------------------------------------------------ #
#  Step 3 & 4 — BFS traversal + subgraph construction                  #
# ------------------------------------------------------------------ #

def _build_adjacency(triples: List[Dict]) -> Dict[str, List[int]]:
    """Build entity -> [triple_indices] adjacency map."""
    adj: Dict[str, List[int]] = defaultdict(list)
    for idx, t in enumerate(triples):
        subj = t.get("subject", "")
        obj = t.get("object", "")
        if subj:
            adj[subj].append(idx)
        if obj:
            adj[obj].append(idx)
    return adj


def _select_connected_subgraph(
    triples: List[Dict],
    seed_scores: Dict[str, float],
    max_triples: int = MAX_SUBGRAPH_TRIPLES,
) -> List[Dict]:
    """BFS from seed entities, selecting triples that form connected reasoning paths.

    Scoring: each triple gets a relevance score based on:
      - seed proximity (hop distance from seed nodes)
      - predicate informativeness (svo_* > co_occurs)
      - connectivity (triples sharing entities with already-selected triples)
    """
    if not triples or not seed_scores:
        return triples[:max_triples]  # fallback: return raw triples

    adj = _build_adjacency(triples)

    # BFS from all seed entities simultaneously
    # Visit entities level by level; for each entity, mark its triples
    triple_scores: Dict[int, float] = {}  # triple_index -> score
    visited_entities: Set[str] = set()
    queue: deque = deque()  # (entity_name, hop, seed_score)

    # Initialise queue with seed entities (sorted by score desc for priority)
    for entity, score in sorted(seed_scores.items(), key=lambda x: -x[1]):
        queue.append((entity, 0, score))

    max_hops = 3  # don't go too far from seeds

    while queue:
        entity, hop, seed_score = queue.popleft()
        if entity in visited_entities:
            continue
        visited_entities.add(entity)

        if hop > max_hops:
            continue

        # Score all triples involving this entity
        hop_decay = 1.0 / (1.0 + hop)
        for tidx in adj.get(entity, []):
            t = triples[tidx]
            pred = t.get("predicate", "")

            # Predicate bonus: SVO triples are more informative than co_occurs
            if pred.startswith("svo_"):
                pred_bonus = 1.5
            elif pred == "co_occurs":
                pred_bonus = 0.5
            else:
                pred_bonus = 1.0

            score = seed_score * hop_decay * pred_bonus
            # Keep the best score if triple seen from multiple paths
            if tidx not in triple_scores or score > triple_scores[tidx]:
                triple_scores[tidx] = score

            # Enqueue the other end of this triple
            other_entity = t.get("object", "") if t.get("subject", "") == entity else t.get("subject", "")
            if other_entity and other_entity not in visited_entities:
                queue.append((other_entity, hop + 1, seed_score))

    if not triple_scores:
        return triples[:max_triples]

    # Sort by score and pick top candidates
    ranked = sorted(triple_scores.items(), key=lambda x: -x[1])

    # Greedily select triples that form a connected graph
    selected_indices: List[int] = []
    selected_entities: Set[str] = set()

    for tidx, score in ranked:
        if len(selected_indices) >= max_triples:
            break

        t = triples[tidx]
        subj = t.get("subject", "")
        obj = t.get("object", "")

        # First triple always accepted; after that prefer connected triples
        if selected_indices:
            connects = (subj in selected_entities) or (obj in selected_entities)
            if not connects and len(selected_indices) >= MIN_SUBGRAPH_TRIPLES:
                continue  # skip disconnected triples once we have enough

        selected_indices.append(tidx)
        if subj:
            selected_entities.add(subj)
        if obj:
            selected_entities.add(obj)

    # Second pass: fill remaining slots with any high-scoring connected triples we skipped
    if len(selected_indices) < max_triples:
        for tidx, score in ranked:
            if tidx in set(selected_indices):
                continue
            if len(selected_indices) >= max_triples:
                break
            t = triples[tidx]
            subj = t.get("subject", "")
            obj = t.get("object", "")
            if (subj in selected_entities) or (obj in selected_entities):
                selected_indices.append(tidx)
                if subj:
                    selected_entities.add(subj)
                if obj:
                    selected_entities.add(obj)

    result = [triples[i] for i in selected_indices]
    logger.info(
        f"Subgraph selector: {len(triples)} raw triples -> {len(result)} selected "
        f"({len(selected_entities)} entities, {len(seed_scores)} seeds)"
    )
    return result


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def select_reasoning_subgraph(
    query: str,
    triples: List[Dict],
    max_triples: int = MAX_SUBGRAPH_TRIPLES,
) -> List[Dict]:
    """Select a small, connected, query-relevant subgraph from raw triples.

    This is the main entry point. Call it between fetching raw triples from
    Neo4j and passing them to the generation prompt.

    Args:
        query: The user's natural language question.
        triples: Raw triples from GraphStore.get_subgraph_triples().
        max_triples: Maximum triples to return (default 10).

    Returns:
        A filtered list of triple dicts forming a connected reasoning subgraph.
    """
    if not triples:
        return []

    if len(triples) <= max_triples:
        logger.debug(f"Subgraph selector: only {len(triples)} triples, returning all")
        return triples

    # Step 1: Identify query concepts
    query_concepts = _extract_query_concepts(query)
    logger.debug(f"Query concepts: {query_concepts}")

    if not query_concepts:
        # Can't identify concepts; return top triples by predicate quality
        sorted_triples = sorted(
            triples,
            key=lambda t: _predicate_quality(t.get("predicate", "")),
            reverse=True,
        )
        return sorted_triples[:max_triples]

    # Step 2: Find seed nodes
    seed_scores = _find_seed_entities(triples, query_concepts)
    logger.debug(f"Seed entities: {list(seed_scores.keys())[:10]}")

    # Steps 3 & 4: BFS traversal + connected subgraph selection
    selected = _select_connected_subgraph(triples, seed_scores, max_triples)

    return selected
