from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
	PyMuPDFLoader,
	Docx2txtLoader,
	TextLoader,
	BSHTMLLoader,
)
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from .config import settings


def _normalize_corpus_record(doc: Document) -> Dict[str, Any]:
	meta = doc.metadata or {}
	return {
		"text": doc.page_content,
		"source": meta.get("source", "unknown"),
		"page": meta.get("page", meta.get("page_number")),
		"chunk_id": meta.get("chunk_id"),
	}


class RAGService:
	def __init__(self) -> None:
		self.llm = ChatOllama(
			model=settings.LLM_MODEL,
			base_url=settings.OLLAMA_BASE_URL,
			temperature=0.1,
			timeout=settings.OLLAMA_REQUEST_TIMEOUT_S,
			keep_alive=settings.OLLAMA_KEEP_ALIVE_S,
		)
		self.embeddings = OllamaEmbeddings(
			model=settings.EMBEDDING_MODEL,
			base_url=settings.OLLAMA_BASE_URL,
			keep_alive=settings.OLLAMA_KEEP_ALIVE_S,
		)
		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=settings.CHUNK_SIZE,
			chunk_overlap=settings.CHUNK_OVERLAP,
			separators=["\n\n", "\n", ". ", ".", " "]
		)
		self.chroma_path = str(settings.CHROMA_DIR)
		Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
		self.collection_name = settings.COLLECTION_NAME
		self.corpus_file = settings.CORPUS_FILE
		self._bm25: BM25Okapi | None = None
		self._bm25_docs: List[Dict[str, Any]] = []
		if self.corpus_file.exists():
			self._load_bm25()

	def _get_vectorstore(self) -> Chroma:
		return Chroma(
			collection_name=self.collection_name,
			embedding_function=self.embeddings,
			persist_directory=self.chroma_path,
		)

	def _vector_count(self) -> int:
		try:
			vs = self._get_vectorstore()
			return vs._collection.count()  # type: ignore[attr-defined]
		except Exception:
			return 0

	def _gather_txt_files(self, root: Path) -> List[Path]:
		files: List[Path] = []
		for p in root.rglob("*.txt"):
			if p.is_file():
				files.append(p)
		return files

	def _load_file(self, file_path: Path) -> List[Document]:
		ext = file_path.suffix.lower()
		if ext == ".pdf":
			loader = PyMuPDFLoader(str(file_path))
		elif ext == ".docx":
			loader = Docx2txtLoader(str(file_path))
		elif ext in {".txt", ".md"}:
			loader = TextLoader(str(file_path), encoding="utf-8")
		elif ext in {".html", ".htm"}:
			loader = BSHTMLLoader(str(file_path))
		else:
			return []
		docs = loader.load()
		# Normalize page metadata when available
		for d in docs:
			m = d.metadata or {}
			m.setdefault("source", str(file_path))
			if "page" in m:
				m["page_number"] = m.get("page")
			d.metadata = m
		return docs

	def _append_corpus(self, docs: List[Document]) -> None:
		self.corpus_file.parent.mkdir(parents=True, exist_ok=True)
		with self.corpus_file.open("a", encoding="utf-8") as f:
			for d in docs:
				record = _normalize_corpus_record(d)
				f.write(json.dumps(record, ensure_ascii=False) + "\n")

	def _load_bm25(self) -> None:
		self._bm25_docs = []
		corpus_texts: List[List[str]] = []
		with self.corpus_file.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				obj = json.loads(line)
				self._bm25_docs.append(obj)
				corpus_texts.append(obj["text"].split())
		self._bm25 = BM25Okapi(corpus_texts) if corpus_texts else None

	def init_on_boot(self) -> dict:
		"""Initialize indexes on boot: load BM25, and if vector store empty, ingest txt files from AILA subdirs."""
		bm25_loaded = False
		if self.corpus_file.exists():
			self._load_bm25()
			bm25_loaded = self._bm25 is not None
		vec_count = self._vector_count()
		ingested = {"files": 0, "chunks": 0}
		try:
			if vec_count == 0 and settings.AILA_DIR.exists():
				cases_dir = settings.AILA_DIR / "Object_casedocs"
				stats_dir = settings.AILA_DIR / "Object_statutes"
				files: List[Path] = []
				if cases_dir.exists():
					files.extend(self._gather_txt_files(cases_dir))
				if stats_dir.exists():
					files.extend(self._gather_txt_files(stats_dir))
				if files:
					_, chunks = self.ingest_files(files)
					ingested = {"files": len(files), "chunks": chunks}
		except Exception:
			pass
		return {
			"bm25_loaded": bm25_loaded,
			"vector_count": self._vector_count(),
			"ingested": ingested,
		}

	def ingest_files(self, file_paths: List[Path]) -> Tuple[int, int]:
		documents: List[Document] = []
		for file_path in file_paths:
			documents.extend(self._load_file(file_path))
		if not documents:
			return 0, 0
		splits = self.text_splitter.split_documents(documents)
		# Assign chunk ids
		for idx, d in enumerate(splits):
			m = d.metadata or {}
			m["chunk_id"] = idx
			d.metadata = m
		# Persist dense
		vs = self._get_vectorstore()
		vs.add_documents(splits)
		vs.persist()
		# Persist corpus for BM25 and load index
		self._append_corpus(splits)
		self._load_bm25()
		return len(documents), len(splits)

	def _dense_search(self, query: str, k: int) -> List[Document]:
		retriever = self._get_vectorstore().as_retriever(search_kwargs={"k": k})
		return retriever.invoke(query)

	def _bm25_search(self, query: str, k: int) -> List[Document]:
		if not self._bm25:
			return []
		tokens = query.split()
		scores = self._bm25.get_scores(tokens)
		idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
		docs: List[Document] = []
		for i in idxs:
			rec = self._bm25_docs[i]
			d = Document(page_content=rec["text"], metadata={"source": rec["source"], "page": rec.get("page"), "chunk_id": rec.get("chunk_id")})
			docs.append(d)
		return docs

	def asearch(self, query: str, k: int | None = None) -> List[Document]:
		k = k or settings.RETRIEVAL_K
		if not settings.USE_HYBRID:
			return self._dense_search(query, k)
		dense_docs = self._dense_search(query, k)
		bm25_docs = self._bm25_search(query, k)
		# Reciprocal Rank Fusion
		ranked: Dict[str, float] = {}
		def key(d: Document) -> str:
			return f"{d.metadata.get('source')}|{d.metadata.get('chunk_id')}"
		for rank, d in enumerate(dense_docs, start=1):
			ranked[key(d)] = ranked.get(key(d), 0.0) + 1.0 / (60 + rank)
		for rank, d in enumerate(bm25_docs, start=1):
			ranked[key(d)] = ranked.get(key(d), 0.0) + 1.0 / (60 + rank)
		# Merge unique by key, preserve best content/metadata from dense
		unique: Dict[str, Document] = {key(d): d for d in dense_docs}
		for d in bm25_docs:
			kkey = key(d)
			if kkey not in unique:
				unique[kkey] = d
		# Sort by fused score
		sorted_keys = sorted(unique.keys(), key=lambda kk: ranked.get(kk, 0.0), reverse=True)
		return [unique[kk] for kk in sorted_keys[:k]]

	def generate_answer(self, query: str, docs: List[Document]) -> str:
		context_blocks = []
		for i, d in enumerate(docs, start=1):
			snippet = d.page_content[:1200]
			source = d.metadata.get("source", "unknown")
			page = d.metadata.get("page") or d.metadata.get("page_number")
			page_tag = f", p.{page}" if page is not None else ""
			context_blocks.append(f"[Doc {i}] Source: {source}{page_tag}\n{snippet}")
		context = "\n\n".join(context_blocks)
		prompt = (
			"You are a legal research assistant. Answer strictly using the cited context. "
			"Quote short spans and include [Doc N] tags inline. If unsupported, say you cannot find it.\n\n"
			f"Question: {query}\n\nContext:\n{context}\n\n"
			"Return a concise answer with inline citations like [Doc 2]."
		)
		resp = self.llm.invoke(prompt)
		return resp

	def get_stats(self) -> dict:
		vs = self._get_vectorstore()
		count = 0
		try:
			count = vs._collection.count()  # type: ignore[attr-defined]
		except Exception:
			return {
			"collection": self.collection_name,
			"persist_directory": self.chroma_path,
			"doc_count": 0,
			"bm25_ready": self._bm25 is not None,
			"corpus_file": str(self.corpus_file),
		}
		bm25_ready = self._bm25 is not None
		return {
			"collection": self.collection_name,
			"persist_directory": self.chroma_path,
			"doc_count": count,
			"bm25_ready": bm25_ready,
			"corpus_file": str(self.corpus_file),
		}
