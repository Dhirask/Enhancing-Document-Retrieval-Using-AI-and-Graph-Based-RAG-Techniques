import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Entity types worth creating co-occurrence edges for (skip noisy numeric types)
MEANINGFUL_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "EVENT", "PRODUCT",
    "WORK_OF_ART", "LAW", "NORP", "FAC", "LOC",
    "LANGUAGE",
}

# Alias resolution: common abbreviations / alternate forms -> canonical ID suffix
_ALIAS_TABLE: Dict[str, str] = {
    "u.s.": "united_states", "u.s.a.": "united_states",
    "usa": "united_states", "us": "united_states",
    "united states of america": "united_states",
    "united states": "united_states",
    "uk": "united_kingdom", "u.k.": "united_kingdom",
    "united kingdom": "united_kingdom",
    "eu": "european_union", "e.u.": "european_union",
    "un": "united_nations", "u.n.": "united_nations",
    "who": "world_health_organization", "w.h.o.": "world_health_organization",
    "nato": "north_atlantic_treaty_organization",
    "ai": "artificial_intelligence",
    "ml": "machine_learning",
    "nlp": "natural_language_processing",
    "llm": "large_language_model", "llms": "large_language_model",
}


@dataclass
class Chunk:
    id: str
    text: str
    source_document: str


@dataclass
class Entity:
    id: str
    label: str
    type: str


@dataclass
class Relation:
    head: str
    tail: str
    type: str
    score: float


@dataclass
class IngestionResult:
    chunks: List[Chunk]
    entities: List[Entity]
    relations: List[Relation]


def _canonical_entity_id(label: str) -> str:
    """Generate a canonical entity ID from the surface form.

    Applies alias resolution, strips leading articles, lowercases,
    and replaces non-alphanumeric chars with underscores.
    """
    text = label.lower().strip()
    # Strip leading articles
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            text = text[len(article):]
    # Alias resolution
    if text in _ALIAS_TABLE:
        return f"ent_{_ALIAS_TABLE[text]}"
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return f"ent_{normalized}" if normalized else "ent_unknown"


class IngestionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover
            raise ImportError("spaCy is required for NER. Install with `pip install spacy` and download a model.") from exc

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as exc:  # pragma: no cover
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run `python -m spacy download en_core_web_sm`."
            ) from exc

    def ingest(self, paths: Iterable[str]) -> IngestionResult:
        documents = self._load_documents(paths)
        logger.info(f"Loaded {len(documents)} documents")

        chunks = self._chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No chunks created from input documents")

        entities, chunk_entities, chunk_docs = self._extract_entities(chunks)
        logger.info(f"Extracted {len(entities)} unique canonical entities")

        relations = self._extract_relations(chunks, entities, chunk_entities, chunk_docs)
        logger.info(f"Extracted {len(relations)} relations")

        return IngestionResult(chunks=chunks, entities=entities, relations=relations)

    # ------------------------------------------------------------------ #
    #  Document Loading                                                    #
    # ------------------------------------------------------------------ #
    def _load_documents(self, paths: Iterable[str]) -> List[str]:
        docs = []
        for path in paths:
            suffix = pathlib.Path(path).suffix.lower()
            if suffix not in self.config.allowed_formats:
                continue
            if suffix == ".pdf":
                docs.append(self._read_pdf(path))
            else:
                docs.append(pathlib.Path(path).read_text(encoding="utf-8"))
        return docs

    def _read_pdf(self, path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pypdf is required to read PDFs. Install with `pip install pypdf`.") from exc

        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)

    # ------------------------------------------------------------------ #
    #  Sentence-Aware Chunking                                             #
    # ------------------------------------------------------------------ #
    def _chunk_documents(self, documents: List[str]) -> List[Chunk]:
        """Split documents into chunks using sentence boundaries via spaCy."""
        chunks: List[Chunk] = []
        for idx, doc_text in enumerate(documents):
            spacy_doc = self._nlp(doc_text)
            sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

            current_words: List[str] = []
            chunk_start_sentence = 0

            for sent_idx, sentence in enumerate(sentences):
                sent_words = sentence.split()
                if current_words and len(current_words) + len(sent_words) > self.config.chunk_size:
                    # Emit current chunk
                    chunk_id = f"chunk_{idx}_{chunk_start_sentence}"
                    chunks.append(Chunk(
                        id=chunk_id,
                        text=" ".join(current_words),
                        source_document=f"doc_{idx}",
                    ))
                    # Overlap: keep last N words
                    if self.config.chunk_overlap > 0 and len(current_words) > self.config.chunk_overlap:
                        current_words = current_words[-self.config.chunk_overlap:]
                    else:
                        current_words = []
                    chunk_start_sentence = sent_idx

                current_words.extend(sent_words)

            # Emit remaining words as final chunk
            if current_words:
                chunk_id = f"chunk_{idx}_{chunk_start_sentence}"
                chunks.append(Chunk(
                    id=chunk_id,
                    text=" ".join(current_words),
                    source_document=f"doc_{idx}",
                ))
        return chunks

    # ------------------------------------------------------------------ #
    #  Entity Extraction with Canonical Deduplication                       #
    # ------------------------------------------------------------------ #
    def _extract_entities(self, chunks: List[Chunk]) -> Tuple[List[Entity], Dict[str, List[str]], Dict[str, object]]:
        """Extract entities using spaCy NER with canonical deduplication.

        Returns:
            entities: Deduplicated list of Entity objects (one per canonical name).
            chunk_entities: Mapping from chunk.id -> list of canonical entity IDs found in that chunk.
            chunk_docs: Mapping from chunk.id -> spaCy Doc (cached for reuse by downstream stages).
        """
        canonical_entities: Dict[str, Entity] = {}  # canonical_id -> Entity
        chunk_entities: Dict[str, List[str]] = {}    # chunk_id -> [canonical_ids]
        chunk_docs: Dict[str, object] = {}           # chunk_id -> spaCy Doc

        for chunk in chunks:
            doc = self._nlp(chunk.text)
            chunk_docs[chunk.id] = doc
            seen_in_chunk: Set[str] = set()
            chunk_entities[chunk.id] = []

            for ent in doc.ents:
                canon_id = _canonical_entity_id(ent.text)
                # Register canonical entity (first occurrence wins for label/type)
                if canon_id not in canonical_entities:
                    canonical_entities[canon_id] = Entity(
                        id=canon_id,
                        label=ent.text.strip(),
                        type=ent.label_,
                    )
                # Link chunk to canonical entity (deduplicated within chunk)
                if canon_id not in seen_in_chunk:
                    chunk_entities[chunk.id].append(canon_id)
                    seen_in_chunk.add(canon_id)

        return list(canonical_entities.values()), chunk_entities, chunk_docs

    # ------------------------------------------------------------------ #
    #  Relation Extraction (structural + co-occurrence + SVO triples)       #
    # ------------------------------------------------------------------ #
    def _extract_relations(
        self, chunks: List[Chunk], entities: List[Entity],
        chunk_entities: Dict[str, List[str]], chunk_docs: Dict[str, object],
    ) -> List[Relation]:
        relations: List[Relation] = []
        entity_lookup = {e.id: e for e in entities}

        for chunk in chunks:
            # Structural: chunk --part_of--> document
            relations.append(Relation(head=chunk.id, tail=chunk.source_document, type="part_of", score=1.0))

            ent_ids = chunk_entities.get(chunk.id, [])

            # Provenance: entity --mentions--> chunk
            for ent_id in ent_ids:
                relations.append(Relation(head=ent_id, tail=chunk.id, type="mentions", score=1.0))

            # ---- Sentence-scoped co-occurrence (meaningful entity types only) ----
            doc = chunk_docs.get(chunk.id) or self._nlp(chunk.text)
            for sent in doc.sents:
                # Collect meaningful entities in this sentence
                sent_ent_ids: List[str] = []
                for ent in sent.ents:
                    if ent.label_ not in MEANINGFUL_ENTITY_TYPES:
                        continue
                    canon_id = _canonical_entity_id(ent.text)
                    if canon_id in entity_lookup:
                        sent_ent_ids.append(canon_id)
                # Deduplicate within sentence
                sent_ent_ids = list(dict.fromkeys(sent_ent_ids))
                if len(sent_ent_ids) > 1:
                    for a_idx in range(len(sent_ent_ids)):
                        for b_idx in range(a_idx + 1, len(sent_ent_ids)):
                            a, b = sent_ent_ids[a_idx], sent_ent_ids[b_idx]
                            if a == b:
                                continue
                            relations.append(Relation(head=a, tail=b, type="co_occurs", score=1.0))
                            relations.append(Relation(head=b, tail=a, type="co_occurs", score=1.0))

            # SVO triples via dependency parsing (verb stored in type)
            svo_rels = self._extract_svo_relations(chunk, entity_lookup, ent_ids, doc)
            relations.extend(svo_rels)

        return relations

    def _extract_svo_relations(
        self, chunk: Chunk, entity_lookup: Dict[str, Entity], ent_ids: List[str],
        doc: object = None,
    ) -> List[Relation]:
        """Extract Subject-Verb-Object triples from chunk text using spaCy dependency parsing.

        Creates typed relation edges between canonical entities that participate in SVO patterns.
        Uses pre-computed spaCy Doc when provided to avoid redundant processing.
        """
        if doc is None:
            doc = self._nlp(chunk.text)
        relations: List[Relation] = []

        # Build a lookup from lowercased entity label -> canonical id (for this chunk's entities)
        label_to_id: Dict[str, str] = {}
        for eid in ent_ids:
            ent = entity_lookup.get(eid)
            if ent:
                label_to_id[ent.label.lower()] = eid

        for token in doc:
            # Find verbs
            if token.pos_ != "VERB":
                continue

            # Collect subjects and objects
            subjects: List[str] = []
            objects: List[str] = []

            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Check if the subject span overlaps with a named entity
                    subj_text = self._get_entity_span_text(child, doc)
                    subj_lower = subj_text.lower()
                    if subj_lower in label_to_id:
                        subjects.append(label_to_id[subj_lower])
                elif child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    obj_text = self._get_entity_span_text(child, doc)
                    obj_lower = obj_text.lower()
                    if obj_lower in label_to_id:
                        objects.append(label_to_id[obj_lower])

            # Also check prepositional objects
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            pobj_text = self._get_entity_span_text(grandchild, doc)
                            pobj_lower = pobj_text.lower()
                            if pobj_lower in label_to_id:
                                objects.append(label_to_id[pobj_lower])

            # Create SVO relations with verb-specific type (e.g. svo_founded, svo_acquired)
            verb_lemma = re.sub(r"[^a-z0-9]", "_", token.lemma_.lower().strip()).strip("_")
            if not verb_lemma:
                continue
            svo_type = f"svo_{verb_lemma}"
            for subj_id in subjects:
                for obj_id in objects:
                    if subj_id != obj_id:
                        relations.append(Relation(
                            head=subj_id,
                            tail=obj_id,
                            type=svo_type,
                            score=1.0,
                        ))
                        logger.debug(
                            f"SVO: {entity_lookup[subj_id].label} --[{svo_type}]--> "
                            f"{entity_lookup[obj_id].label} (chunk {chunk.id})"
                        )

        return relations

    @staticmethod
    def _get_entity_span_text(token, doc) -> str:
        """Get the entity text that a token belongs to, or the token's own text."""
        for ent in doc.ents:
            if ent.start <= token.i < ent.end:
                return ent.text
        # Fallback: try the subtree (compound nouns)
        subtree = sorted(token.subtree, key=lambda t: t.i)
        return " ".join(t.text for t in subtree)
