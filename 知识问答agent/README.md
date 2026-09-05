# 知识问答 Agent — 智能客服系统

一个从零手写的 RAG（检索增强生成）知识问答系统：企业/部门维护自己的知识库，用户在 Web 聊天页面提问，系统通过 **混合检索 + Rerank 精排** 找到最相关的知识片段，由大模型流式生成准确答案和解决步骤。

## 项目路线图（与 note/test.md 知识点对应）

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 手写 RAG 全链路 + FastAPI 服务化 + Web 聊天前端（SSE 流式） | ✅ 完成 |
| Phase 1 | BM25+向量混合检索（RRF 融合）+ BGE Reranker 精排 | ✅ 完成 |
| Phase 2 | 构建"问题-答案-上下文"测试集，接入 Ragas 评测 | 待开始 |
| Phase 3 | 语义缓存 + 兜底降级 + 反馈闭环 | 待开发 |
| Phase 4 | Function Calling 查工单/实时数据，ReAct 处理多步问题 | 待开发 |
| Phase 5 | Docker 容器化、vLLM 部署、LangFuse 可观测性 | 待开发 |

## 架构

```
Web 前端 (static/)  ──SSE──►  FastAPI (server.py)
                                  │
data/knowledge/*.md               ▼  RAGPipeline (kbqa/pipeline.py)
        │                   ┌──────────────────────────────┐
        ▼  loader.py        │ HybridRetriever (retriever.py)
        ▼  chunker.py       │   ├─ 向量检索 Top-20 (store.py + embedder.py)
        ▼  embedder.py      │   ├─ BM25 关键词检索 (bm25.py, jieba 分词)
        ▼  store.py         │   ├─ 加权 RRF 融合
   Chroma 向量库             │   └─ BGE Reranker 精排 (reranker.py)
                            │ generator.py  内网 LLM + 约束性 System Prompt
                            └──────────────────────────────┘
                                  ▼
                        流式答案 + 参考来源
```

### 检索链路（Phase 1）

1. **双路召回**：向量检索（语义相似，bge-small-zh）+ BM25（关键词精确匹配，jieba 分词），各取 Top-20 候选。
2. **RRF 融合**（Reciprocal Rank Fusion）：`score = Σ wᵢ / (k + rankᵢ)`，权重可配（默认向量 0.6 / BM25 0.4，k=60）。BM25 补足了纯向量检索对"退款到账时间"这类精确关键词的短板。
3. **Rerank 精排**：CrossEncoder（bge-reranker-base）对 Query-文档逐对打分，比双塔模型精度更高，压缩到 Top-3 送入 LLM，并按 `rerank_threshold` 过滤不相关结果。

可在 `kbqa/config.py` 中用 `use_hybrid` / `use_rerank` 开关做消融对比。

## 快速开始

### 0. 环境要求

- Python 3.11（已验证 3.11.8）
- vLLM 服务（任意 OpenAI 兼容接口均可）

### 1. 创建虚拟环境并安装依赖

```bash
cd 知识问答agent
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install rank_bm25 jieba     # Phase 1 检索依赖
```

### 2. 下载模型（仅需一次）

> 公司内网代理会拦截 huggingface.co / hf-mirror.com。请切换到有外网的环境（如手机热点）执行，模型会缓存到项目 `data/models/` 目录，之后离线可用。

```bash
python scripts/download_models.py --endpoint https://hf-mirror.com
```

仅需下载 **1 个模型**（约 100MB）：
- `BAAI/bge-small-zh-v1.5` — 中文 Embedding（内网 LLM 服务不提供 embeddings 接口，向量检索必须本地运行）

> Rerank 无需下载模型：已改为内网 LLM 打分实现。

### 3. LLM：使用内网模型服务（无需自建）

项目已接入公司内网推理服务（OpenAI 兼容接口），开箱即用：

- 服务地址：`https://ai-models.rnd.yinwang.com/ai-platform-models-agent/v1`
- 可选模型：`GLM-5.3-Flash`（默认，快）、`DeepSeek-V4-Flash-0731`、`GLM-5.3`（强）
- Web 页面右上角下拉框可随时切换模型；CLI 用 `--model` 参数指定

### 4. 修改配置（按需）

编辑 `kbqa/config.py`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `api_base` | 内网 LLM 服务地址 | `https://ai-models.rnd.yinwang.com/...` |
| `api_key` | 内网服务 API Key | 已配置 |
| `llm_models` | 前端可切换的模型列表 | GLM-5.3-Flash / DeepSeek-V4-Flash-0731 / GLM-5.3 |
| `llm_model` | 默认模型 | `GLM-5.3-Flash` |
| `verify_ssl` | 内网自签名证书需为 `False` | `False` |
| `embedding_model` | 本地 `data/models/` 有则优先用本地 | `BAAI/bge-small-zh-v1.5` |
| `server_host` / `server_port` | Web 服务监听地址 | `0.0.0.0:8080` |
| `chunk_size` / `chunk_overlap` | 切分长度/重叠 | 400 / 60 |
| `top_k` / `score_threshold` | 最终召回数 / 向量相似度阈值 | 5 / 0.30 |
| `use_hybrid` | 是否启用混合检索 | `True` |
| `vector_weight` / `bm25_weight` | RRF 融合权重 | 0.6 / 0.4 |
| `use_rerank` | 是否启用 Rerank 精排 | `True` |
| `rerank_backend` | 精排方式：`llm`（内网 LLM 打分）| `llm` |
| `rerank_top_n` / `rerank_threshold` | 送入 LLM 的文档数 / 精排分数阈值 | 3 / 3.0 |

### 5. 构建知识库

```bash
python cli.py ingest
```

### 6. 启动 Web 服务

```bash
python server.py
```

浏览器访问 **http://localhost:8080**，在聊天页面提问。页面右上角可切换模型，顶部状态栏显示知识库条数和检索模式。

### 调试模式（可选）

```bash
python cli.py chat                              # 终端交互式问答
python cli.py ask "怎么退货？"                   # 单次提问（默认模型）
python cli.py ask "怎么退货？" --model GLM-5.3   # 指定模型
```

## 项目结构

```
知识问答agent/
├── server.py                FastAPI 服务：/api/chat（SSE 流式）、/api/health、静态页托管
├── cli.py                   命令行调试入口（ingest / chat / ask）
├── requirements.txt         依赖清单
├── README.md
│
├── kbqa/                    核心代码包
│   ├── config.py            全局配置（检索参数、LLM 地址、模型路径解析）
│   ├── loader.py            文档解析：读取 data/knowledge/ 下 Markdown/TXT
│   ├── chunker.py           文本切分：按 Markdown 标题→段落→句子递归切分，带 overlap
│   ├── embedder.py          Embedding 封装：BGE 中文模型，查询加指令前缀，懒加载
│   ├── store.py             Chroma 向量库封装：持久化、cosine 相似度、Top-K 检索
│   ├── bm25.py              BM25 索引：jieba 搜索引擎分词 + rank_bm25
│   ├── reranker.py          Rerank 精排：内网 LLM 批量打分（0-10），JSON 解析 + 失败回退 RRF 顺序
│   ├── retriever.py         检索器：Retriever（纯向量基线）/ HybridRetriever（双路召回+RRF 融合+Rerank）
│   ├── generator.py         LLM 生成：内网服务(OpenAI 兼容)，防幻觉 System Prompt，支持流式与模型切换
│   └── pipeline.py          RAG 全链路编排：ingest（建库）/ query / stream_query
│
├── static/                  Web 前端（原生 HTML/CSS/JS，无框架）
│   ├── index.html           聊天页面骨架（含模型切换下拉框）
│   ├── style.css            样式（仿微信客服界面）
│   └── app.js               前端逻辑：SSE 流式渲染、模型切换、快捷问题、状态检查、防重复提交
│
├── scripts/
│   └── download_models.py   模型预下载脚本（在有外网的环境执行一次）
│
└── data/
    ├── knowledge/           知识库文档（Markdown，当前 5 篇示例：退货/保修/账号/支付/物流）
    ├── models/              本地模型缓存（download_models.py 生成）
    └── chroma/              向量库持久化目录（ingest 自动生成）
```

## 前端功能

- SSE 流式渲染，打字机效果，首字延迟低
- 状态栏自检：知识库为空 / 服务未启动时给出明确操作提示
- 常见问题快捷入口，点击即问
- Enter 发送、Shift+Enter 换行、发送中禁用输入防重复提交

## 设计要点

- **先手写后框架**：MVP 不用 LangChain，每个环节独立成模块，便于理解 RAG 全链路；后续用 LangChain/LlamaIndex 重构对比。
- **防幻觉三道闸**：检索阈值过滤（rerank_threshold）→ System Prompt 强约束"无依据则答不知道" → 回答强制标注参考来源。
- **BGE 查询前缀**：查询时使用 BGE 官方指令前缀，缩小 Query 与文档的语义差距（HyDE 思想的轻量替代）。
- **双路召回互补**：向量检索管语义（"钱什么时候回来"→退款文档），BM25 管精确词（"发票""运费险"等专有名词）。
- **离线友好**：模型自动优先加载 `data/models/` 本地目录，适配公司内网环境。
