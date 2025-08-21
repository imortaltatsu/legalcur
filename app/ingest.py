from __future__ import annotations

from pathlib import Path
from typing import List
import zipfile
import io
import requests

from .config import settings
from .rag import RAGService


AILA_ZENODO_URL = "https://zenodo.org/records/4063986/files/AILA_2019_dataset.zip?download=1"


def download_aila_dataset(target_dir: Path) -> List[Path]:
	target_dir.mkdir(parents=True, exist_ok=True)
	resp = requests.get(AILA_ZENODO_URL, timeout=120)
	resp.raise_for_status()
	z = zipfile.ZipFile(io.BytesIO(resp.content))
	extracted_paths: List[Path] = []
	for name in z.namelist():
		if name.endswith("/"):
			continue
		out_path = target_dir / name
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with z.open(name) as src, open(out_path, "wb") as dst:
			dst.write(src.read())
		extracted_paths.append(out_path)
	return extracted_paths


def gather_files_from_dir(root: Path) -> List[Path]:
	exts = {".pdf", ".txt", ".docx", ".html"}
	files: List[Path] = []
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in exts:
			files.append(p)
	return files


def ingest_aila() -> dict:
	# Prefer local dataset if present
	root = settings.AILA_DIR
	files: List[Path]
	if root.exists():
		files = gather_files_from_dir(root)
	else:
		data_root = settings.DATA_DIR / "aila2019"
		files = download_aila_dataset(data_root)
	service = RAGService()
	_, chunks = service.ingest_files(files)
	return {"files": len(files), "chunks": chunks}


def ingest_local_dir(path: str) -> dict:
	root = Path(path)
	if not root.exists():
		raise FileNotFoundError(f"Path not found: {path}")
	files = gather_files_from_dir(root)
	service = RAGService()
	_, chunks = service.ingest_files(files)
	return {"files": len(files), "chunks": chunks}
