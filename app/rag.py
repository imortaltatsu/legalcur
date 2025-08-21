from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from .config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
	def __init__(self) -> None:
		logger.info("Initializing RAG Service...")
		
		# Initialize Ollama clients
		self.llm = ChatOllama(
			model=settings.LLM_MODEL,
			base_url=settings.OLLAMA_BASE_URL,
			temperature=0.1,
			timeout=settings.OLLAMA_REQUEST_TIMEOUT_S,
		)
		
		self.embeddings = OllamaEmbeddings(
			model=settings.EMBEDDING_MODEL,
			base_url=settings.OLLAMA_BASE_URL,
		)
		
		# Text processing
		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=settings.CHUNK_SIZE,
			chunk_overlap=settings.CHUNK_OVERLAP,
			separators=["\n\n", "\n", ". ", ".", " "]
		)
		
		# Storage paths
		self.chroma_path = settings.CHROMA_DIR
		self.chroma_path.mkdir(parents=True, exist_ok=True)
		
		self.corpus_file = settings.CORPUS_FILE
		self.corpus_file.parent.mkdir(parents=True, exist_ok=True)
		
		self.collection_name = settings.COLLECTION_NAME
		
		# BM25 components
		self._bm25: BM25Okapi | None = None
		self._bm25_docs: List[Dict[str, Any]] = []
		
		# Load existing data if available
		self._load_existing_data()
		
		logger.info("RAG Service initialized successfully")

	def _load_existing_data(self) -> None:
		"""Load existing BM25 index and corpus if available."""
		if self.corpus_file.exists():
			logger.info(f"Loading existing corpus from {self.corpus_file}")
			self._load_bm25()
		else:
			logger.info("No existing corpus found")

	def _get_vectorstore(self) -> Chroma:
		"""Get or create Chroma vector store."""
		return Chroma(
			collection_name=self.collection_name,
			embedding_function=self.embeddings,
			persist_directory=str(self.chroma_path),
		)

	def _load_txt_files(self, directory: Path) -> List[Path]:
		"""Load all .txt files from a directory recursively."""
		if not directory.exists():
			logger.warning(f"Directory {directory} does not exist")
			return []
		
		txt_files = list(directory.rglob("*.txt"))
		logger.info(f"Found {len(txt_files)} .txt files in {directory}")
		return txt_files

	def _load_document(self, file_path: Path) -> List[Document]:
		"""Load a single text document."""
		try:
			loader = TextLoader(str(file_path), encoding="utf-8")
			docs = loader.load()
			
			# Add metadata
			for doc in docs:
				doc.metadata.update({
					"source": str(file_path),
					"file_type": "txt",
					"file_name": file_path.name,
				})
			
			return docs
		except Exception as e:
			logger.error(f"Error loading {file_path}: {e}")
			return []

	def _save_corpus(self, docs: List[Document]) -> None:
		"""Save documents to corpus file."""
		try:
			with self.corpus_file.open("w", encoding="utf-8") as f:
				for doc in docs:
					record = {
						"text": doc.page_content,
						"source": doc.metadata.get("source", "unknown"),
						"metadata": doc.metadata,
					}
					f.write(json.dumps(record, ensure_ascii=False) + "\n")
			logger.info(f"Saved {len(docs)} documents to corpus")
		except Exception as e:
			logger.error(f"Error saving corpus: {e}")

	def _load_bm25(self) -> None:
		"""Load BM25 index from corpus file."""
		try:
			self._bm25_docs = []
			corpus_texts = []
			
			with self.corpus_file.open("r", encoding="utf-8") as f:
				for line_num, line in enumerate(f, 1):
					line = line.strip()
					if not line:
						continue
					
					try:
						obj = json.loads(line)
						self._bm25_docs.append(obj)
						corpus_texts.append(obj["text"].split())
					except json.JSONDecodeError as e:
						logger.warning(f"Invalid JSON at line {line_num}: {e}")
			
			if corpus_texts:
				self._bm25 = BM25Okapi(corpus_texts)
				logger.info(f"Loaded BM25 index with {len(corpus_texts)} documents")
			else:
				logger.warning("No valid documents found in corpus")
				
		except Exception as e:
			logger.error(f"Error loading BM25: {e}")
			self._bm25 = None

	def _get_document_count(self) -> int:
		"""Get document count from vector store."""
		try:
			vs = self._get_vectorstore()
			# Try different methods to get count
			if hasattr(vs, '_collection') and vs._collection:
				return vs._collection.count()
			elif hasattr(vs, 'collection') and vs.collection:
				return vs.collection.count()
			else:
				# Fallback: count from corpus
				if self.corpus_file.exists():
					with self.corpus_file.open("r", encoding="utf-8") as f:
						return sum(1 for line in f if line.strip())
				return 0
		except Exception as e:
			logger.error(f"Error getting document count: {e}")
			return 0

	def init_on_boot(self) -> dict:
		"""Initialize indexes on boot."""
		logger.info("Starting boot initialization...")
		
		# Check current state
		current_count = self._get_document_count()
		bm25_ready = self._bm25 is not None
		
		ingested = {"files": 0, "chunks": 0}
		
		# If no documents exist, ingest from AILA dataset
		if current_count == 0:
			logger.info("No documents found, starting ingestion...")
			ingested = self._ingest_aila_dataset()
		
		# Reload BM25 after potential ingestion
		if not bm25_ready:
			self._load_bm25()
			bm25_ready = self._bm25 is not None
		
		final_count = self._get_document_count()
		
		result = {
			"bm25_ready": bm25_ready,
			"initial_count": current_count,
			"final_count": final_count,
			"ingested": ingested,
			"corpus_file": str(self.corpus_file),
			"chroma_dir": str(self.chroma_path),
		}
		
		logger.info(f"Boot initialization complete: {result}")
		return result

	def _ingest_aila_dataset(self) -> Dict[str, int]:
		"""Ingest documents from AILA dataset directories."""
		logger.info("Ingesting AILA dataset...")
		
		all_files = []
		
		# Load from cases directory
		if settings.CASES_DIR.exists():
			case_files = self._load_txt_files(settings.CASES_DIR)
			all_files.extend(case_files)
			logger.info(f"Found {len(case_files)} case files")
		
		# Load from statutes directory
		if settings.STATUTES_DIR.exists():
			statute_files = self._load_txt_files(settings.STATUTES_DIR)
			all_files.extend(statute_files)
			logger.info(f"Found {len(statute_files)} statute files")
		
		if not all_files:
			logger.warning("No files found in AILA dataset directories")
			return {"files": 0, "chunks": 0}
		
		# Process all files
		all_docs = []
		for file_path in all_files:
			docs = self._load_document(file_path)
			all_docs.extend(docs)
		
		if not all_docs:
			logger.warning("No documents loaded from files")
			return {"files": 0, "chunks": 0}
		
		# Split documents into chunks
		chunks = self.text_splitter.split_documents(all_docs)
		logger.info(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
		
		# Add to vector store
		try:
			vs = self._get_vectorstore()
			vs.add_documents(chunks)
			vs.persist()
			logger.info("Documents added to vector store")
		except Exception as e:
			logger.error(f"Error adding to vector store: {e}")
		
		# Save to corpus for BM25
		self._save_corpus(chunks)
		
		# Build BM25 index
		self._load_bm25()
		
		return {"files": len(all_files), "chunks": len(chunks)}

	def search(self, query: str, k: int | None = None) -> List[Document]:
		"""Search for relevant documents using hybrid retrieval."""
		k = k or settings.RETRIEVAL_K
		logger.info(f"Searching for query: '{query}' with k={k}")
		
		# Get dense search results
		dense_docs = self._dense_search(query, k)
		logger.info(f"Dense search returned {len(dense_docs)} documents")
		
		# Get BM25 search results
		bm25_docs = self._bm25_search(query, k)
		logger.info(f"BM25 search returned {len(bm25_docs)} documents")
		
		if not settings.USE_HYBRID:
			return dense_docs
		
		# Combine results using reciprocal rank fusion
		combined_docs = self._combine_results(dense_docs, bm25_docs, k)
		logger.info(f"Combined search returned {len(combined_docs)} documents")
		
		return combined_docs

	def _dense_search(self, query: str, k: int) -> List[Document]:
		"""Perform dense vector search."""
		try:
			vs = self._get_vectorstore()
			retriever = vs.as_retriever(search_kwargs={"k": k})
			return retriever.invoke(query)
		except Exception as e:
			logger.error(f"Error in dense search: {e}")
			return []

	def _bm25_search(self, query: str, k: int) -> List[Document]:
		"""Perform BM25 search."""
		if not self._bm25:
			return []
		
		try:
			tokens = query.split()
			scores = self._bm25.get_scores(tokens)
			
			# Get top k results
			top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
			
			docs = []
			for idx in top_indices:
				if idx < len(self._bm25_docs):
					record = self._bm25_docs[idx]
					doc = Document(
						page_content=record["text"],
						metadata=record.get("metadata", {})
					)
					docs.append(doc)
			
			return docs
		except Exception as e:
			logger.error(f"Error in BM25 search: {e}")
			return []

	def _combine_results(self, dense_docs: List[Document], bm25_docs: List[Document], k: int) -> List[Document]:
		"""Combine dense and BM25 results using reciprocal rank fusion."""
		# Create a mapping of documents by source
		doc_map = {}
		
		# Add dense results
		for rank, doc in enumerate(dense_docs, 1):
			key = doc.metadata.get("source", str(id(doc)))
			doc_map[key] = {
				"doc": doc,
				"dense_score": 1.0 / (60 + rank),
				"bm25_score": 0.0
			}
		
		# Add BM25 results
		for rank, doc in enumerate(bm25_docs, 1):
			key = doc.metadata.get("source", str(id(doc)))
			if key in doc_map:
				doc_map[key]["bm25_score"] = 1.0 / (60 + rank)
			else:
				doc_map[key] = {
					"doc": doc,
					"dense_score": 0.0,
					"bm25_score": 1.0 / (60 + rank)
				}
		
		# Sort by combined score
		sorted_docs = sorted(
			doc_map.values(),
			key=lambda x: x["dense_score"] + x["bm25_score"],
			reverse=True
		)
		
		return [item["doc"] for item in sorted_docs[:k]]

	def generate_answer(self, query: str, docs: List[Document]) -> str:
		"""Generate an answer using retrieved documents."""
		if not docs:
			return "I couldn't find any relevant documents to answer your question."
		
		# Prepare context
		context_blocks = []
		for i, doc in enumerate(docs, 1):
			snippet = doc.page_content[:1200]
			source = doc.metadata.get("source", "unknown")
			file_name = doc.metadata.get("file_name", "unknown")
			
			context_blocks.append(f"[Doc {i}] Source: {file_name}\n{snippet}")
		
		context = "\n\n".join(context_blocks)
		
		# Create prompt
		prompt = f"""You are a legal research assistant. Answer the question using only the provided context.

Question: {query}

Context:
{context}

Instructions:
- Answer based ONLY on the provided context
- Include [Doc N] citations for any information you use
- If the context doesn't contain enough information, say so
- Be concise and accurate

Answer:"""
		
		try:
			response = self.llm.invoke(prompt)
			return response
		except Exception as e:
			logger.error(f"Error generating answer: {e}")
			return f"Error generating answer: {e}"

	def get_stats(self) -> dict:
		"""Get current system statistics."""
		doc_count = self._get_document_count()
		bm25_ready = self._bm25 is not None
		
		return {
			"collection": self.collection_name,
			"persist_directory": str(self.chroma_path),
			"doc_count": doc_count,
			"bm25_ready": bm25_ready,
			"corpus_file": str(self.corpus_file),
			"corpus_exists": self.corpus_file.exists(),
			"corpus_size": self.corpus_file.stat().st_size if self.corpus_file.exists() else 0,
			"cases_dir": str(settings.CASES_DIR),
			"statutes_dir": str(settings.STATUTES_DIR),
		}
