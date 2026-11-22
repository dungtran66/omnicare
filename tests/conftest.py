"""Unit test luôn chạy MockLLM (deterministic, không tốn lượt API) kể cả khi .env cấu hình model thật."""
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_MODEL"] = ""  # RAG dùng char 3-gram, không gọi API embedding
import tempfile, time  # noqa: E401
os.environ["TRACE_DB"] = os.path.join(tempfile.gettempdir(), f"oc_pytest_{time.time_ns()}.db")  # không ghi vào DB thật
