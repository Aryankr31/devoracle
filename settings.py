from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")

    # GitHub
    github_token: str = Field(default="", env="GITHUB_TOKEN")
    github_target_repo: str = Field(default="", env="GITHUB_TARGET_REPO")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = "devoracle_codebase"

    # App
    app_env: str = Field(default="development", env="APP_ENV")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    api_port: int = Field(default=8000, env="API_PORT")

    # Chunking
    chunk_size: int = 1000          # tokens per chunk
    chunk_overlap: int = 150        # overlap between chunks
    max_files_per_ingest: int = 500 # safety limit per run

    # Retrieval
    retrieval_top_k: int = 8        # docs to retrieve per query

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
