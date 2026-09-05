from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    db_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "kbqa"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 5
    score_threshold: float = 0.30

    use_hybrid: bool = True
    hybrid_candidates: int = 20
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60

    use_rerank: bool = True
    rerank_backend: str = "llm"  # llm | off
    rerank_top_n: int = 3
    rerank_threshold: float = 3.0

    api_base: str = "https://ai-models.rnd.yinwang.com/ai-platform-models-agent/v1"
    api_key: str = "d5227d4d6a014c45b615c36b9c023b20"
    llm_models: list = field(
        default_factory=lambda: ["GLM-5.3-Flash", "DeepSeek-V4-Flash-0731", "GLM-5.3"]
    )
    llm_model: str = "GLM-5.3-Flash"
    verify_ssl: bool = False
    temperature: float = 0.2
    max_tokens: int = 1024

    server_host: str = "0.0.0.0"
    server_port: int = 8080

    models_dir: Path = PROJECT_ROOT / "data" / "models"


def resolve_model(name: str) -> str:
    """本地模型目录中存在同名目录时优先使用本地路径（离线环境）。"""
    local = CONFIG.models_dir / name.split("/")[-1]
    if local.exists():
        return str(local)
    return name


CONFIG = Config()
