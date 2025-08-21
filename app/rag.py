from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
	def __init__(self) -> None:
		logger.info("Initializing RAG Service...")
		
		# Initialize Ollama client for LLM
		self.llm = ChatOllama(
			model=settings.LLM_MODEL,
			base_url=settings.OLLAMA_BASE_URL,
			temperature=0.1,
			timeout=settings.OLLAMA_REQUEST_TIMEOUT_S,
		)
		
		# Initialize local embedding model using transformers
		self.embeddings = HuggingFaceEmbeddings(
			model_name=settings.EMBEDDING_MODEL,
			model_kwargs={'device': 'cpu'},  # Use CPU by default, can be changed to 'cuda' if available
			encode_kwargs={'normalize_embeddings': True}
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

	def _get_master_prompt(self, query_type: str = "general") -> str:
		"""Get the master prompt for legal assistance."""
		base_prompt = """You are an expert legal research assistant with comprehensive knowledge of legal systems, statutes, case law, and legal procedures. You provide accurate, well-reasoned legal analysis based on the provided context.

CORE CAPABILITIES:
- Legal Research & Analysis
- Statute Interpretation
- Case Law Analysis
- Legal Procedure Guidance
- Contract Review & Analysis
- Regulatory Compliance
- Legal Document Drafting
- Risk Assessment

RESPONSE GUIDELINES:
1. **Accuracy First**: Base all responses on the provided legal context
2. **Citation Required**: Always cite specific documents using [Doc N] format
3. **Legal Precision**: Use precise legal terminology and concepts
4. **Practical Application**: Provide actionable legal advice when possible
5. **Risk Awareness**: Highlight potential legal risks and limitations
6. **Professional Tone**: Maintain professional, objective legal analysis
7. **Context Limitations**: Acknowledge when context is insufficient

CONTEXT USAGE:
- Analyze provided legal documents thoroughly
- Cross-reference statutes with case law when available
- Identify key legal principles and precedents
- Note any conflicts or ambiguities in the law
- Suggest relevant legal considerations

DISCLAIMER: This analysis is based on the provided context and should not replace professional legal counsel. Users should consult qualified attorneys for specific legal advice."""

		# Specialized prompts for different query types
		specialized_prompts = {
			"statute": """SPECIALIZATION: STATUTE ANALYSIS
Focus on:
- Statutory interpretation principles
- Legislative intent analysis
- Statutory construction rules
- Amendment history and implications
- Regulatory framework connections""",
			
			"case": """SPECIALIZATION: CASE LAW ANALYSIS
Focus on:
- Precedent establishment and application
- Fact pattern analysis
- Legal reasoning and holdings
- Dissenting opinions and implications
- Case law evolution and trends""",
			
			"procedure": """SPECIALIZATION: LEGAL PROCEDURE
Focus on:
- Procedural requirements and timelines
- Jurisdictional considerations
- Filing requirements and forms
- Evidence standards and admissibility
- Appeal processes and deadlines""",
			
			"contract": """SPECIALIZATION: CONTRACT LAW
Focus on:
- Contract formation and validity
- Terms interpretation and enforcement
- Breach analysis and remedies
- Risk allocation and liability
- Regulatory compliance requirements""",
			
			"compliance": """SPECIALIZATION: REGULATORY COMPLIANCE
Focus on:
- Regulatory framework analysis
- Compliance requirements and deadlines
- Enforcement mechanisms and penalties
- Risk assessment and mitigation
- Industry-specific considerations"""
		}
		
		specialized = specialized_prompts.get(query_type, "")
		if specialized:
			base_prompt += f"\n\n{specialized}"
		
		return base_prompt

	def _analyze_query_type(self, query: str) -> str:
		"""Analyze query to determine the type of legal assistance needed."""
		query_lower = query.lower()
		
		# Statute-related keywords
		if any(word in query_lower for word in ["statute", "law", "act", "code", "regulation", "ordinance", "legislation"]):
			return "statute"
		
		# Case law keywords
		if any(word in query_lower for word in ["case", "precedent", "ruling", "decision", "judgment", "court", "appeal"]):
			return "case"
		
		# Procedure keywords
		if any(word in query_lower for word in ["procedure", "process", "filing", "deadline", "jurisdiction", "venue", "service"]):
			return "procedure"
		
		# Contract keywords
		if any(word in query_lower for word in ["contract", "agreement", "breach", "enforcement", "terms", "liability"]):
			return "contract"
		
		# Compliance keywords
		if any(word in query_lower for word in ["compliance", "regulatory", "enforcement", "penalty", "violation", "audit"]):
			return "compliance"
		
		return "general"

	def _format_context_for_prompt(self, docs: List[Document]) -> str:
		"""Format retrieved documents into a structured context for the prompt."""
		if not docs:
			return "No relevant legal documents found in the provided context."
		
		context_blocks = []
		for i, doc in enumerate(docs, 1):
			# Extract key information
			source = doc.metadata.get("source", "unknown")
			file_name = doc.metadata.get("file_name", "unknown")
			
			# Clean and format content
			content = doc.page_content.strip()
			if len(content) > 1500:
				content = content[:1500] + "..."
			
			# Create structured block
			block = f"""[Document {i}]
Source: {file_name}
Path: {source}
Content:
{content}
---"""
			context_blocks.append(block)
		
		return "\n\n".join(context_blocks)

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
				for doc in tqdm(docs, desc="Saving to corpus", unit="doc"):
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
		
		# Process all files with progress bar
		logger.info("Loading documents from files...")
		all_docs = []
		for file_path in tqdm(all_files, desc="Loading files", unit="file"):
			docs = self._load_document(file_path)
			all_docs.extend(docs)
		
		if not all_docs:
			logger.warning("No documents loaded from files")
			return {"files": 0, "chunks": 0}
		
		# Split documents into chunks with progress bar
		logger.info("Splitting documents into chunks...")
		chunks = self.text_splitter.split_documents(all_docs)
		logger.info(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
		
		# Add to vector store with progress bar
		try:
			logger.info("Adding documents to vector store...")
			vs = self._get_vectorstore()
			
			# Process in batches for better progress tracking
			batch_size = 100
			for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector store", unit="batch"):
				batch = chunks[i:i + batch_size]
				vs.add_documents(batch)
			
			vs.persist()
			logger.info("Documents added to vector store")
		except Exception as e:
			logger.error(f"Error adding to vector store: {e}")
		
		# Save to corpus for BM25
		logger.info("Saving documents to corpus...")
		self._save_corpus(chunks)
		
		# Build BM25 index
		logger.info("Building BM25 index...")
		self._load_bm25()
		
		return {"files": len(all_files), "chunks": len(chunks)}

	def ingest_files(self, files: List[Path]) -> Tuple[int, int]:
		"""Ingest a list of files and return (file_count, chunk_count)."""
		logger.info(f"Ingesting {len(files)} files...")
		
		if not files:
			logger.warning("No files provided for ingestion")
			return 0, 0
		
		# Load documents from files with progress bar
		logger.info("Loading documents from files...")
		all_docs = []
		for file_path in tqdm(files, desc="Loading files", unit="file"):
			docs = self._load_document(file_path)
			all_docs.extend(docs)
		
		if not all_docs:
			logger.warning("No documents loaded from files")
			return len(files), 0
		
		# Split documents into chunks with progress bar
		logger.info("Splitting documents into chunks...")
		chunks = self.text_splitter.split_documents(all_docs)
		logger.info(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
		
		# Add to vector store with progress bar
		try:
			logger.info("Adding documents to vector store...")
			vs = self._get_vectorstore()
			
			# Process in batches for better progress tracking
			batch_size = 100
			for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector store", unit="batch"):
				batch = chunks[i:i + batch_size]
				vs.add_documents(batch)
			
			vs.persist()
			logger.info("Documents added to vector store")
		except Exception as e:
			logger.error(f"Error adding to vector store: {e}")
		
		# Save to corpus for BM25
		logger.info("Saving documents to corpus...")
		self._save_corpus(chunks)
		
		# Build BM25 index
		logger.info("Building BM25 index...")
		self._load_bm25()
		
		return len(files), len(chunks)

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
		"""Generate a comprehensive legal answer using retrieved documents."""
		if not docs:
			return "I couldn't find any relevant legal documents to answer your question. Please provide more specific details or consult with a qualified legal professional."
		
		# Analyze query type for specialized prompting
		query_type = self._analyze_query_type(query)
		logger.info(f"Query type detected: {query_type}")
		
		# Get master prompt
		master_prompt = self._get_master_prompt(query_type)
		
		# Format context
		context = self._format_context_for_prompt(docs)
		
		# Create comprehensive prompt
		prompt = f"""{master_prompt}

LEGAL QUERY:
{query}

RELEVANT LEGAL CONTEXT:
{context}

ANALYSIS INSTRUCTIONS:
1. **Comprehensive Analysis**: Provide a thorough legal analysis based on the context
2. **Statutory Framework**: Identify and explain relevant statutes and regulations
3. **Case Law Application**: Apply relevant case law and precedents
4. **Legal Principles**: Extract and explain key legal principles
5. **Practical Implications**: Discuss practical legal implications and considerations
6. **Risk Assessment**: Identify potential legal risks and limitations
7. **Citation Format**: Use [Doc N] citations throughout your analysis
8. **Professional Tone**: Maintain objective, professional legal analysis

Please provide a comprehensive legal analysis addressing the query above."""

		try:
			response = self.llm.invoke(prompt)
			return response
		except Exception as e:
			logger.error(f"Error generating answer: {e}")
			return f"Error generating legal analysis: {e}. Please try again or consult with a legal professional."

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
