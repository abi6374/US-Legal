from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from models.clause_detection import detect_clauses
from models.citation_verification import extract_and_verify_citations
from models.ai_content_detection import detect_ai_content
from models.faq_chat import get_legal_response
from models.summarization import summarize_text
from fastapi import UploadFile, File as FastAPIFile, Form
from models.rag_pdf import answer_from_upload
from pydantic import BaseModel
import os
import tempfile
from models.courtlistener import get_case_from_courtlistener
from typing import List

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    print("Working")
    return {"message": "Welcome to the Legal Query AI API"}

class InputModel(BaseModel):
    task: str
    text: str
    file: Optional[str] = None
    objective: Optional[str] = None
    focus: Optional[str] = None
    length: Optional[str] = None
    context: Optional[str] = None

@app.post("/analyze")
async def analyze(input: InputModel):
    response: Dict[str, Any] = {
        "task": input.task,
        "text": input.text,
        "context": input.context
    }
    print(response)
    try:
        if input.task == "contract_clause_detection":
            result = detect_clauses(input.text)
        elif input.task == "citation_verification":
            result = extract_and_verify_citations(input.text)
        elif input.task == "ai_content_detection":
            result = detect_ai_content(input.text)
        elif input.task == "faq_chat":
            result = get_legal_response(input.text, input.context or "")
        elif input.task == "summarization":
            result = summarize_text(input.text, input.focus or "", input.length or "medium")
        else:
            result = {"error": "Invalid or unsupported task type."}

        if isinstance(result, dict):
            for k, v in result.items():
                response[k] = v
        else:
            response["error"] = "Unexpected result format."

    except Exception as e:
        print(e)
        response["error"] = f"Processing error: {str(e)}"

    return JSONResponse(response)


# Direct endpoint for Summarization.jsx
from fastapi import UploadFile, File as FastAPIFile
class SummarizeModel(BaseModel):
    text: str
    focus: Optional[str] = None
    length: Optional[str] = None

@app.post("/summarize")
async def summarize(input: SummarizeModel):
    result = summarize_text(input.text, input.focus or "", input.length or "medium")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"summary": result["summary"]}

# Dedicated endpoint for FAQ Chat (accepts 'question' or 'text')
class FAQChatModel(BaseModel):
    question: Optional[str] = None
    text: Optional[str] = None
    context: Optional[str] = None


class ClauseDetectModel(BaseModel):
    text: str

@app.post("/clause_detection")
async def clause_detection(input: ClauseDetectModel):
    result = detect_clauses(input.text)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/faq_chat")
async def faq_chat(input: FAQChatModel):
    q = input.question or input.text
    if not q:
        raise HTTPException(status_code=400, detail="Provide 'question' or 'text' in the body.")
    return get_legal_response(q, input.context or "")



class RAGPDFResponse(BaseModel):
    response: str
    sources: list

@app.post("/rag_pdf/qa")
async def rag_pdf_qa(
    file: UploadFile = FastAPIFile(...),
    question: Optional[str] = Form(None),   # from form-data
    q: Optional[str] = None,                # fallback via query ?q=...
    chunk_size: Optional[int] = Form(800),
    chunk_overlap: Optional[int] = Form(120),
):
    try:
        question_val = question or q
        if not question_val:
            raise HTTPException(status_code=400, detail="Provide 'question' as form field or '?q=' query.")
        file_bytes = await file.read()
        result = answer_from_upload(
            file_bytes=file_bytes,
            filename=file.filename,
            question=question_val,
            chunk_size=chunk_size or 800,
            chunk_overlap=chunk_overlap or 120
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
# in main.py (add imports at top)


class CourtLookupModel(BaseModel):
    citation_text: str

class CourtLookupBatchModel(BaseModel):
    citations: List[str]

@app.post("/courtlistener/lookup")
async def courtlistener_lookup(input: CourtLookupModel):
    result = get_case_from_courtlistener(input.citation_text)
    if not result:
        raise HTTPException(status_code=404, detail="No result found for the given citation.")
    return result

@app.post("/courtlistener/lookup_batch")
async def courtlistener_lookup_batch(input: CourtLookupBatchModel):
    out = []
    for c in input.citations:
        r = get_case_from_courtlistener(c)
        out.append({"citation": c, "result": r})
    return {"results": out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)