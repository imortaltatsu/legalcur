from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
	# Ollama setup
j	OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
	OLLAMA_REQUEST_TIMEOUT_S: int = Field(default=300)
	OLLAMA_KEEP_ALIVE_S: int = Field(default=300)

	# Models (chat + embeddings via Ollama)
	LLM_MODEL: str = Field(default="gpt-oss:120b")
	EMBEDDING_MODEL: str = Field(default="mxbai-embed-large")

	# Storage and data - relative paths
	DATA_DIR: Path = Field(default=Path("data"))
	AILA_DIR: Path = Field(default=Path("data/AILA_2019_dataset"))
	CASES_DIR: Path = Field(default=Path("data/AILA_2019_dataset/Object_casedocs"))
	STATUTES_DIR: Path = Field(default=Path("data/AILA_2019_dataset/Object_statutes"))
	CHROMA_DIR: Path = Field(default=Path("storage/chroma"))
	COLLECTION_NAME: str = Field(default="aila")
	CORPUS_FILE: Path = Field(default=Path("storage/corpus.jsonl"))

	# RAG defaults
	RETRIEVAL_K: int = Field(default=5)
	CHUNK_SIZE: int = Field(default=800)
	CHUNK_OVERLAP: int = Field(default=150)
	USE_HYBRID: bool = Field(default=True)

	class Config:
		env_file = ".env"
		extra = "ignore"


settings = Settings()
