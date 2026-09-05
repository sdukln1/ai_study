from __future__ import annotations

import time

from kbqa.chunker import chunk_text
from kbqa.config import CONFIG
from kbqa.embedder import Embedder
from kbqa.generator import Generator
from kbqa.loader import load_documents
from kbqa.logger import get_logger
from kbqa.retriever import HybridRetriever, Retriever
from kbqa.store import VectorStore

logger = get_logger("pipeline")


class RAGPipeline:
    def __init__(self):
        start = time.time()
        self.embedder = Embedder()
        self.store = VectorStore()
        if CONFIG.use_hybrid:
            self.retriever = HybridRetriever(self.embedder, self.store)
        else:
            self.retriever = Retriever(self.embedder, self.store)
        self.generator = None
        logger.info(
            f"RAGPipeline 初始化完成 耗时={time.time() - start:.2f}s "
            f"模式={'混合检索' if CONFIG.use_hybrid else '纯向量'} "
            f"rerank={'开' if CONFIG.use_rerank else '关'} "
            f"知识库块数={self.store.count()}"
        )

    def _get_generator(self) -> Generator:
        if self.generator is None:
            self.generator = Generator()
        return self.generator

    def ingest(self) -> int:
        documents = load_documents()
        if not documents:
            logger.warning("知识库目录为空，请先在 data/knowledge 中放入文档")
            return 0

        all_chunks = []
        for doc in documents:
            for chunk in chunk_text(doc.text):
                all_chunks.append({"text": chunk, "metadata": doc.metadata})
        logger.info(f"[ingest] 加载 {len(documents)} 个文档，切分出 {len(all_chunks)} 个文本块")

        logger.info(f"[ingest] 使用 {CONFIG.embedding_model} 生成向量...")
        start = time.time()
        embeddings = self.embedder.embed_documents([c["text"] for c in all_chunks])
        logger.info(f"[ingest] 向量生成完成 耗时={time.time() - start:.2f}s")
        added = self.store.add(all_chunks, embeddings)
        logger.info(f"[ingest] 已写入向量库，当前共 {self.store.count()} 个文本块")
        if CONFIG.use_hybrid and hasattr(self.retriever, "rebuild_bm25"):
            self.retriever.rebuild_bm25()
            logger.info("[ingest] BM25 索引已重建")
        return added

    def query(self, question: str, model: str | None = None) -> dict:
        logger.info(f"[query] 收到提问 model={model or CONFIG.llm_model} question={question!r}")
        start = time.time()
        hits = self.retriever.retrieve(question)
        logger.info(f"[query] 检索完成 耗时={time.time() - start:.2f}s 命中 {len(hits)} 条")
        context = Retriever.format_context(hits)
        generator = self._get_generator()
        answer = generator.generate(question, context, model=model)
        logger.info(f"[query] 全链路完成 总耗时={time.time() - start:.2f}s")
        return {"answer": answer, "sources": hits}

    def stream_query(self, question: str, model: str | None = None):
        logger.info(f"[stream] 收到提问 model={model or CONFIG.llm_model} question={question!r}")
        start = time.time()
        hits = self.retriever.retrieve(question)
        logger.info(
            f"[stream] 检索完成 耗时={time.time() - start:.2f}s 命中 {len(hits)} 条: "
            + str([(h['metadata'].get('source'), round(h['score'], 2)) for h in hits])
        )
        context = Retriever.format_context(hits)
        if not hits:
            logger.warning("[stream] 检索无命中，返回兜底话术")
            yield "抱歉，知识库中暂时没有找到相关信息，建议您联系人工客服。"
            return
        generator = self._get_generator()
        for token in generator.stream_generate(question, context, model=model):
            yield token
        logger.info(f"[stream] 问答全链路完成 总耗时={time.time() - start:.2f}s")
        yield "\n" + self._sources_line(hits)

    @staticmethod
    def _sources_line(hits: list[dict]) -> str:
        sources = sorted({h["metadata"].get("source", "unknown") for h in hits})
        return "参考来源: " + "、".join(sources)
