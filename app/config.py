"""Cấu hình OmniCare, đọc từ biến môi trường (hoặc file .env)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _load_dotenv() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# --- Model ---
# LLM_PROVIDER = "openai" (mọi API tương thích OpenAI: OpenAI, Groq, OpenRouter, vLLM...), "ollama" hoặc "mock" (offline).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_PRIMARY_TIMEOUT = float(os.getenv("LLM_PRIMARY_TIMEOUT", "0"))
# Số lượt gọi model tối đa cho mỗi khách (mỗi phiên chat) trong một ngày; 0 = không giới hạn
MODEL_CALLS_PER_DAY = int(os.getenv("MODEL_CALLS_PER_DAY", "40"))
LLM_FALLBACK_MODELS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]

# --- Claude (LLM_PROVIDER=anthropic) ---
# Khóa: ANTHROPIC_API_KEY (hoặc LLM_API_KEY). Model mặc định Claude Opus 5.5; đổi qua LLM_MODEL,
# vd claude-sonnet-5-5 hoặc claude-haiku-5-5 nếu muốn rẻ hơn.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")  # low | medium | high | xhigh | max

# --- RAG ---
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "2.0"))  # điểm BM25 tối thiểu để coi là có căn cứ

# --- Embedding (tùy chọn, API tương thích OpenAI /embeddings; để trống = dùng char 3-gram offline) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")          # vd: text-embedding-3-small, bge-m3, multilingual-e5-large
# Anthropic (Claude) không có API /embeddings, nên không lấy mặc định theo LLM_BASE_URL khi đang dùng Claude.
_llm_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1" if "anthropic.com" in _llm_url else _llm_url)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "" if "anthropic.com" in _llm_url else os.getenv("LLM_API_KEY", ""))

# --- Thương mại ---
FREESHIP_THRESHOLD = int(os.getenv("FREESHIP_THRESHOLD", "500000"))

# --- Observability ---
TRACE_DB = Path(os.getenv("TRACE_DB", str(BASE_DIR / "omnicare_traces.db")))
