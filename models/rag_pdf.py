import os
import tempfile
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

LEGAL_RAG_SYSTEM_PROMPT = """You are a US legal RAG assistant answering strictly from the provided PDF context.
Rules:
- Answer ONLY if the question is about US law AND the answer is supported by the PDF context.
- Be concise and professional (≤120 words).
- If the PDF does not contain enough information, say: "I cannot answer from the provided document."
- If the question is not legal or not about US law, reply: "I only assist with US legal questions."
- Do not invent facts. Cite page numbers when possible (e.g., "See p. {page}").

Question: {question}
Context:
{context}
"""

def _get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    # Gemini embeddings
    return GoogleGenerativeAIEmbeddings(model="embedding-001")

def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192"
    )

def _split_docs(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def _build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append({
            "page": meta.get("page", None),
            "source": meta.get("source", None),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else "")
        })
    return out

def _make_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template=LEGAL_RAG_SYSTEM_PROMPT
    )

def _build_retrieval_chain(vs: FAISS):
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = _get_llm()
    prompt = _make_prompt()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )
    return chain

def _load_pdf_robust(pdf_path: str) -> List[Document]:
    # Try PyPDF -> PyMuPDF -> PDFPlumber
    errors = []
    try:
        return PyPDFLoader(pdf_path).load()
    except Exception as e:
        errors.append(f"PyPDFLoader: {e}")
    try:
        return PyMuPDFLoader(pdf_path).load()
    except Exception as e:
        errors.append(f"PyMuPDFLoader: {e}")
    try:
        return PDFPlumberLoader(pdf_path).load()
    except Exception as e:
        errors.append(f"PDFPlumberLoader: {e}")
    raise RuntimeError("; ".join(errors))

def answer_from_pdf_path(pdf_path: str, question: str, chunk_size: int = 800, chunk_overlap: int = 120) -> Dict[str, Any]:
    if not question or not question.strip():
        return {"error": "Question cannot be empty"}
    try:
        docs = _load_pdf_robust(pdf_path)
        chunks = _split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vs = _build_vectorstore(chunks)
        chain = _build_retrieval_chain(vs)
        result = chain({"query": question})
        answer = result.get("result", "").strip()
        src_docs = result.get("source_documents", []) or []
        sources = _format_sources(src_docs)
        return {"response": answer, "sources": sources}
    except Exception as e:
        return {"error": f"RAG error: {str(e)}"}

def answer_from_upload(file_bytes: bytes, filename: str, question: str, chunk_size: int = 800, chunk_overlap: int = 120) -> Dict[str, Any]:
    if not file_bytes:
        return {"error": "No file content provided"}
    tmp_path = None
    try:
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        suffix = os.path.splitext(filename)[-1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        return answer_from_pdf_path(tmp_path, question, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        return {"error": f"RAG error: {str(e)}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass