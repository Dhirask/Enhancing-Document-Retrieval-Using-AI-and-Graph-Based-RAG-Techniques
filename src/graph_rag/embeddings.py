from __future__ import annotations

from typing import Iterable, List, Sequence


class TextEmbedder:
    """SentenceTransformer-backed text encoder."""

    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "sentence-transformers is required for TextEmbedder. Install with `pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vectors.tolist()


class GraphEmbedder:
    """Node2Vec graph embedder using NetworkX walks + gensim model."""

    def __init__(
        self,
        dim: int = 128,
        walk_length: int = 20,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 2,
    ) -> None:
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self._model = None
        self._fit_nodes: List[str] = []

    def fit(self, graph: "nx.Graph") -> None:
        try:
            from networkx.algorithms.node2vec import Node2Vec
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("networkx>=2.6 is required for GraphEmbedder Node2Vec.") from exc

        node2vec = Node2Vec(
            graph,
            dimensions=self.dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            quiet=True,
        )
        self._model = node2vec.fit(window=10, min_count=1, batch_words=128)
        self._fit_nodes = list(graph.nodes())

    def encode_nodes(self, node_ids: Iterable[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("GraphEmbedder must be fit on a graph before encoding nodes.")

        vectors: List[List[float]] = []
        for node_id in node_ids:
            if node_id not in self._model.wv:  # type: ignore[attr-defined]
                vectors.append([0.0] * self.dim)
                continue
            vectors.append(self._model.wv[node_id].tolist())  # type: ignore[attr-defined]
        return vectors
