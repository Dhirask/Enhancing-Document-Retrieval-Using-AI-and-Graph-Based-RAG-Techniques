import pathlib
from dataclasses import dataclass
from typing import Iterable, List

from .config import PipelineConfig


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


class IngestionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def ingest(self, paths: Iterable[str]) -> IngestionResult:
        documents = self._load_documents(paths)
        chunks = self._chunk_documents(documents)
        entities = self._extract_entities(chunks)
        relations = self._extract_relations(chunks, entities)
        return IngestionResult(chunks=chunks, entities=entities, relations=relations)

    def _load_documents(self, paths: Iterable[str]) -> List[str]:
        docs = []
        for path in paths:
            suffix = pathlib.Path(path).suffix.lower()
            if suffix not in self.config.allowed_formats:
                continue
            docs.append(pathlib.Path(path).read_text(encoding="utf-8"))
        return docs

    def _chunk_documents(self, documents: List[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for idx, doc in enumerate(documents):
            words = doc.split()
            step = self.config.chunk_size - self.config.chunk_overlap
            for start in range(0, len(words), step):
                piece = words[start : start + self.config.chunk_size]
                chunk_id = f"chunk_{idx}_{start}"
                chunks.append(Chunk(id=chunk_id, text=" ".join(piece), source_document=f"doc_{idx}"))
        return chunks

    def _extract_entities(self, chunks: List[Chunk]) -> List[Entity]:
        entities: List[Entity] = []
        for chunk in chunks:
            # Placeholder: swap with spaCy/transformer-based NER
            if len(chunk.text) > 0:
                entities.append(Entity(id=f"ent_{chunk.id}", label=chunk.text.split()[0], type="MISC"))
        return entities

    def _extract_relations(self, chunks: List[Chunk], entities: List[Entity]) -> List[Relation]:
        relations: List[Relation] = []
        # Placeholder: a real system would apply RE models; here we link each entity to its chunk and document
        for entity in entities:
            relations.append(Relation(head=entity.id, tail=entity.id.replace("ent_", "chunk_"), type="mentions", score=1.0))
        for chunk in chunks:
            relations.append(Relation(head=chunk.id, tail=chunk.source_document, type="part_of", score=1.0))
        return relations
