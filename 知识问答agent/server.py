from __future__ import annotations

import json
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from kbqa.config import CONFIG
from kbqa.logger import get_logger
from kbqa.pipeline import RAGPipeline

logger = get_logger("server")

app = FastAPI(title="智能客服知识问答")

logger.info("正在启动服务...")
pipeline = RAGPipeline()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "kb_chunks": pipeline.store.count(),
        "llm_model": CONFIG.llm_model,
        "llm_models": CONFIG.llm_models,
        "api_base": CONFIG.api_base,
        "hybrid": CONFIG.use_hybrid,
        "rerank": CONFIG.use_rerank,
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    model = body.get("model") or None
    if not question:
        return {"error": "question is required"}
    logger.info(f"[api/chat] 收到请求 model={model or CONFIG.llm_model} question={question!r}")

    def event_stream():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        try:
            for token in pipeline.stream_query(question, model=model):
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[api/chat] 处理异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'服务异常: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


def main():
    import uvicorn

    uvicorn.run(app, host=CONFIG.server_host, port=CONFIG.server_port)


if __name__ == "__main__":
    main()
