from __future__ import annotations

from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag import RAGService
from .config import settings
from .production_config import prod_settings
from langchain_ollama import ChatOllama


app = FastAPI(
    title="Law RAG API", 
    version="0.1.4",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=prod_settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag_service = RAGService()


def get_rag_service() -> RAGService:
	return _rag_service


@app.on_event("startup")
async def on_startup():
	init_info = _rag_service.init_on_boot()
	app.state.rag_init = init_info


class QueryRequest(BaseModel):
	query: str
	k: Optional[int] = None


class ChatMessage(BaseModel):
	role: str  # "system" | "user" | "assistant"
	content: str


class ChatRequest(BaseModel):
	messages: List[ChatMessage]
	k: Optional[int] = None


class EmbedRequest(BaseModel):
	text: str


class ChatDiagRequest(BaseModel):
	prompt: str


@app.get("/")
async def root():
	return {
		"message": "Law RAG API",
		"version": "0.1.4",
		"docs": "/docs",
		"health": "/health"
	}

@app.get("/health")
async def health():
	return {"status": "ok"}


@app.get("/init")
async def get_init_status():
	return getattr(app.state, "rag_init", {})


@app.get("/config")
async def view_config():
	return {
		"ollama_base_url": settings.OLLAMA_BASE_URL,
		"llm_model": settings.LLM_MODEL,
		"embedding_model": settings.EMBEDDING_MODEL,
		"collection": settings.COLLECTION_NAME,
		"chroma_dir": str(settings.CHROMA_DIR),
		"cases_dir": str(settings.CASES_DIR),
		"statutes_dir": str(settings.STATUTES_DIR),
	}


@app.post("/query")
async def query_endpoint(req: QueryRequest, rag: RAGService = Depends(get_rag_service)):
	docs = rag.search(req.query, k=req.k)
	answer = rag.generate_answer(req.query, docs)
	sources = [d.metadata.get("source", "unknown") for d in docs]
	return {"answer": answer, "sources": sources}


@app.post("/search")
async def search_only(req: QueryRequest, rag: RAGService = Depends(get_rag_service)):
	docs = rag.search(req.query, k=req.k)
	return {
		"matches": [
			{"source": d.metadata.get("source", "unknown"), "preview": d.page_content[:240]}
			for d in docs
		]
	}


@app.get("/stats")
async def stats(rag: RAGService = Depends(get_rag_service)):
	return rag.get_stats()


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, rag: RAGService = Depends(get_rag_service)):
	if not req.messages:
		raise HTTPException(status_code=400, detail="messages cannot be empty")
	user_messages = [m for m in req.messages if m.role == "user"]
	if not user_messages:
		raise HTTPException(status_code=400, detail="at least one user message required")
	last_user = user_messages[-1].content

	# Retrieve context based on the last user message
	docs = rag.search(last_user, k=req.k)
	context_blocks = []
	for i, d in enumerate(docs, start=1):
		snippet = d.page_content[:1200]
		source = d.metadata.get("source", "unknown")
		context_blocks.append(f"[Doc {i}] Source: {source}\n{snippet}")
	context = "\n\n".join(context_blocks)

	# Flatten conversation
	conv_lines: List[str] = []
	for m in req.messages:
		role = m.role
		content = m.content
		conv_lines.append(f"{role.upper()}: {content}")
	conversation = "\n".join(conv_lines)

	prompt = (
		"You are a legal research assistant. Use the retrieved context to ground your replies. "
		"Quote short spans and include [Doc N] tags inline. If unsupported, say you cannot find it.\n\n"
		f"Conversation so far:\n{conversation}\n\n"
		f"Retrieved context:\n{context}\n\n"
		"Respond to the last USER message with a concise, cited answer."
	)

	llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.1, timeout=settings.OLLAMA_REQUEST_TIMEOUT_S)
	answer = llm.invoke(prompt)
	sources = [d.metadata.get("source", "unknown") for d in docs]
	return {"answer": answer, "sources": sources}


@app.post("/diag/embeddings")
async def diag_embeddings(req: EmbedRequest, rag: RAGService = Depends(get_rag_service)):
	vec = rag.embeddings.embed_query(req.text)
	return {"dim": len(vec), "preview": vec[:8]}


@app.post("/diag/chat")
async def diag_chat(req: ChatDiagRequest):
	llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.1, timeout=settings.OLLAMA_REQUEST_TIMEOUT_S)
	out = llm.invoke(req.prompt)
	return {"output": out}
