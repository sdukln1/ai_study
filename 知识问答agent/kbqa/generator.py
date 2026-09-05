from __future__ import annotations

import time

import httpx

from kbqa.config import CONFIG
from kbqa.logger import get_logger

logger = get_logger("generator")

SYSTEM_PROMPT = """你是一家电商公司的智能客服助手。请严格遵守以下规则：

1. 只依据下方提供的【参考资料】回答用户问题，不要使用资料之外的知识。
2. 如果参考资料不足以回答问题，直接说"抱歉，知识库中暂时没有找到相关信息，建议您联系人工客服"，不要推测和编造。
3. 回答要简洁、准确、口语化，直接给出解决方案；如涉及流程，请分步骤说明。
4. 一次回答只解决用户当前的问题，不要主动展开无关内容。"""

USER_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请根据参考资料回答用户问题。"""

# 内网服务直连即可；不读代理环境变量，避免本机代理工具劫持内网请求
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)


def create_client():
    import openai

    logger.info(f"初始化 LLM 客户端: {CONFIG.api_base} (trust_env=False)")
    return openai.OpenAI(
        base_url=CONFIG.api_base,
        api_key=CONFIG.api_key,
        max_retries=1,
        http_client=httpx.Client(
            verify=CONFIG.verify_ssl,
            trust_env=False,
            timeout=REQUEST_TIMEOUT,
        ),
    )


def normalize_model(model: str | None) -> str:
    """匹配内网模型名（大小写不敏感）。"""
    if not model:
        return CONFIG.llm_model
    for m in CONFIG.llm_models:
        if m.lower() == model.lower():
            return m
    return model


class Generator:
    def __init__(self):
        self.client = create_client()

    def _build_messages(self, question: str, context: str) -> list[dict]:
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(self, question: str, context: str, model: str | None = None) -> str:
        resolved = normalize_model(model)
        logger.info(f"[LLM] 开始生成 model={resolved} question={question!r} context_len={len(context)}")
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=resolved,
                messages=self._build_messages(question, context),
                temperature=CONFIG.temperature,
                max_tokens=CONFIG.max_tokens,
            )
            answer = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            logger.info(
                f"[LLM] 生成完成 耗时={time.time() - start:.2f}s "
                f"answer_len={len(answer or '')} "
                f"tokens={getattr(usage, 'total_tokens', None) if usage else None}"
            )
            logger.debug(f"[LLM] 完整回答: {answer}")
            return answer
        except Exception as e:
            logger.error(f"[LLM] 生成失败 耗时={time.time() - start:.2f}s error={type(e).__name__}: {e}")
            raise

    def stream_generate(self, question: str, context: str, model: str | None = None):
        resolved = normalize_model(model)
        logger.info(f"[LLM] 开始流式生成 model={resolved} question={question!r} context_len={len(context)}")
        start = time.time()
        first_token_at = None
        total_chars = 0
        try:
            stream = self.client.chat.completions.create(
                model=resolved,
                messages=self._build_messages(question, context),
                temperature=CONFIG.temperature,
                max_tokens=CONFIG.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first_token_at is None:
                        first_token_at = time.time()
                        logger.info(f"[LLM] 首字延迟 {first_token_at - start:.2f}s")
                    total_chars += len(delta.content)
                    logger.debug(f"[LLM] token: {delta.content!r}")
                    yield delta.content
            logger.info(
                f"[LLM] 流式生成完成 总耗时={time.time() - start:.2f}s 输出字符数={total_chars}"
            )
        except Exception as e:
            logger.error(f"[LLM] 流式生成失败 耗时={time.time() - start:.2f}s error={type(e).__name__}: {e}")
            raise
