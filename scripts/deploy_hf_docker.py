"""Đưa backend FastAPI thật (Admin + trang bán hàng, cùng 1 link, đồng bộ qua WebSocket) lên Hugging Face Docker Space.

Khóa model KHÔNG nằm trong mã nguồn tải lên: script đọc các biến LLM_*/EMBEDDING_* từ .env
rồi lưu thành "Secret" của Space (chỉ server đọc được, người xem trang không thấy).

Cách dùng:  python scripts/deploy_hf_docker.py        (đọc HF_TOKEN từ .env hoặc biến môi trường)
Biến tùy chọn: HF_DOCKER_SPACE (mặc định omnicare-live)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


SECRET_KEYS = ["LLM_API_KEY", "EMBEDDING_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "CHATWOOT_API_TOKEN"]
VAR_KEYS = ["LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODELS", "LLM_TIMEOUT", "LLM_PRIMARY_TIMEOUT",
            "LLM_EFFORT", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "SHOP", "FREESHIP_THRESHOLD", "MODEL_CALLS_PER_DAY"]
IGNORE = [".env", ".env.*", "*.db", "*.db-*", "backups/**", ".hf_static_build/**", "data/admin_users.json",
          "tests/**", "**/__pycache__/**", "*.pyc", ".pytest_cache/**", ".git/**", "docs/**", "render.yaml",
          "docker-compose*.yml", "chatwoot.env.example", "README.md", "Dockerfile"]

README = """---
title: OmniCare AI Agent (Live)
emoji: 🛍️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

OmniCare AI Agent — Admin CRM tại `/`, trang bán hàng tại `/store`.
"""

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod -R 777 /app
ENV PORT=7860 TRACE_DB=/tmp/omnicare_traces.db ADMIN_USERS_FILE=/tmp/admin_users.json
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
"""


def main() -> None:
    env = read_env()
    token = os.getenv("HF_TOKEN") or env.get("HF_TOKEN", "")
    if not token:
        sys.exit("Thiếu HF_TOKEN (thêm dòng HF_TOKEN=... vào .env).")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/{os.getenv('HF_DOCKER_SPACE', env.get('HF_DOCKER_SPACE', 'omnicare-live'))}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)

    for k in SECRET_KEYS:
        if env.get(k):
            api.add_space_secret(repo, k, env[k])
    for k in VAR_KEYS:
        if env.get(k):
            api.add_space_variable(repo, k, env[k])
        else:
            try:
                api.delete_space_variable(repo, k)
            except Exception:
                pass

    api.upload_folder(folder_path=str(ROOT), repo_id=repo, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy OmniCare backend")
    api.upload_file(path_or_fileobj=README.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="space")
    api.upload_file(path_or_fileobj=DOCKERFILE.encode("utf-8"), path_in_repo="Dockerfile", repo_id=repo, repo_type="space")
    host = repo.replace("/", "-").lower().replace("_", "-")
    print(f"Space: https://huggingface.co/spaces/{repo}")
    print(f"Admin: https://{host}.hf.space/   Bán hàng: https://{host}.hf.space/store")


if __name__ == "__main__":
    main()
