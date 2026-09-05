from __future__ import annotations

from kbqa.config import CONFIG
from kbqa.embedder import Embedder
from kbqa.store import VectorStore


class Retriever:
    """纯向量检索（Phase 0 基线，用于对比实验）。"""

    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def retrieve(self, question: str) -> list[dict]:
        query_embedding = self.embedder.embed_query(question)
        hits = self.store.search(query_embedding)
        return [h for h in hits if h["score"] >= CONFIG.score_threshold]

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        if not hits:
            return ""
        blocks = []
        for i, hit in enumerate(hits, 1):
            source = hit["metadata"].get("source", "unknown")
            score = hit["score"]
            blocks.append(f"[{i}] (来源: {source}, 相似度: {score:.2f})\n{hit['text']}")
        return "\n\n".join(blocks)


class HybridRetriever:
    """向量检索 + BM25 关键词检索 → 加权 RRF 融合 → Rerank 精排。"""

    def __init__(self, embedder=None, store=None, reranker=None):
        from kbqa.bm25 import BM25Index

        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.bm25 = BM25Index()
        self._reranker = reranker
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        data = self.store.collection.get(include=["documents", "metadatas"])
        chunks = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        self.bm25.build(chunks)

    @property
    def reranker(self):
        if self._reranker is None:
            from kbqa.reranker import LLMReranker

            self._reranker = LLMReranker()
        return self._reranker

    def retrieve(self, question: str) -> list[dict]:
        query_embedding = self.embedder.embed_query(question)
        vec_hits = self.store.search(query_embedding, CONFIG.hybrid_candidates)
        for h in vec_hits:
            h["vector_score"] = h["score"]
        bm_hits = self.bm25.search(question, CONFIG.hybrid_candidates)

        fused = self._rrf_fuse(vec_hits, bm_hits)

        if CONFIG.use_rerank:
            fused = self.reranker.rerank(question, fused)
            return [h for h in fused if h["score"] >= CONFIG.rerank_threshold]
        return fused[: CONFIG.top_k]

    def _rrf_fuse(self, vec_hits: list[dict], bm_hits: list[dict]) -> list[dict]:
        fused: dict[str, dict] = {}

        for rank, hit in enumerate(vec_hits, 1):
            key = hit["text"]
            entry = fused.setdefault(key, {**hit, "rrf": 0.0})
            entry["rrf"] += CONFIG.vector_weight / (CONFIG.rrf_k + rank)

        for rank, hit in enumerate(bm_hits, 1):
            key = hit["text"]
            entry = fused.setdefault(key, {**hit, "rrf": 0.0})
            entry["rrf"] += CONFIG.bm25_weight / (CONFIG.rrf_k + rank)

        results = list(fused.values())
        for r in results:
            r["score"] = r["rrf"]
        results.sort(key=lambda h: h["score"], reverse=True)
        return results
