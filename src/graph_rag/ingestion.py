import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .config import PipelineConfig

logger = logging.getLogger(__name__)


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
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("spaCy is required for NER. Install with `pip install spacy` and download a model.") from exc

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as exc:  # pragma: no cover - model guard
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
        
        entities, chunk_entities = self._extract_entities(chunks)
        logger.info(f"Extracted {len(entities)} unique entities")
        
        relations = self._extract_relations(chunks, entities, chunk_entities)
        logger.info(f"Extracted {len(relations)} relations")
        
        return IngestionResult(chunks=chunks, entities=entities, relations=relations)

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
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("pypdf is required to read PDFs. Install with `pip install pypdf`.") from exc

        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)

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

    def _extract_entities(self, chunks: List[Chunk]) -> Tuple[List[Entity], Dict[str, List[str]]]:
        entities: List[Entity] = []
        chunk_entities: Dict[str, List[str]] = {}
        for chunk in chunks:
            doc = self._nlp(chunk.text)
            chunk_entities[chunk.id] = []
            for i, ent in enumerate(doc.ents):
                ent_id = f"ent_{chunk.id}_{i}"
                entities.append(Entity(id=ent_id, label=ent.text, type=ent.label_))
                chunk_entities[chunk.id].append(ent_id)
        return entities, chunk_entities

    def _extract_relations(
        self, chunks: List[Chunk], entities: List[Entity], chunk_entities: Dict[str, List[str]]
    ) -> List[Relation]:
        relations: List[Relation] = []
        entity_lookup = {e.id: e for e in entities}

        for chunk in chunks:
            relations.append(Relation(head=chunk.id, tail=chunk.source_document, type="part_of", score=1.0))

            ent_ids = chunk_entities.get(chunk.id, [])
            for ent_id in ent_ids:
                relations.append(Relation(head=ent_id, tail=chunk.id, type="mentions", score=1.0))

            if len(ent_ids) > 1:
                for a_idx in range(len(ent_ids)):
                    for b_idx in range(a_idx + 1, len(ent_ids)):
                        a = ent_ids[a_idx]
                        b = ent_ids[b_idx]
                        head_label = entity_lookup[a].label.lower()
                        tail_label = entity_lookup[b].label.lower()
                        score = 1.0 if head_label != tail_label else 0.5
                        relations.append(Relation(head=a, tail=b, type="co_occurs", score=score))
                        relations.append(Relation(head=b, tail=a, type="co_occurs", score=score))

        return relations
