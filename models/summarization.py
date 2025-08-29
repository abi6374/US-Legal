# models/summarization.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

load_dotenv()

SUM_PROMPT = """You are a legal summarization assistant.
Summarize the provided text in a professional, concise manner.

Requirements:
- Focus: {focus}
- Length: {length_label}
- Use clear plain language.
- No fluff. Preserve key legal elements (parties, issues, standards, holdings, obligations) when relevant.
- Avoid adding unsupported facts.

Text:
{text}
"""

def _get_model() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192",
    )

def _length_label(length: str) -> str:
    l = (length or "medium").lower()
    if l == "short":
        return "about 60–90 words"
    if l == "long":
        return "about 200–250 words"
    return "about 120–150 words"

def _create_chain() -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["text", "focus", "length_label"],
        template=SUM_PROMPT,
    )
    return LLMChain(llm=_get_model(), prompt=prompt)

def summarize_text(text: str, focus: str = "", length: str = "medium") -> Dict[str, Any]:
    if not text:
        return {"error": "Text cannot be empty"}
    try:
        chain = _create_chain()
        summary = chain.run(
            text=text,
            focus=focus or "General",
            length_label=_length_label(length),
        ).strip()
        return {"summary": summary}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}