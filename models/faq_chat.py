import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompt for US legal context
LEGAL_SYSTEM_PROMPT = """You are a US legal expert assistant. Answer strictly about US law.
If the question is not about US law, respond exactly: "I only assist with US legal questions."
If attorney consultation is required, state it briefly.

Provide a concise, professional answer:
- Direct answer in 1–2 sentences.
- Key points: 3–5 short bullet points.
- Cite controlling sources if known (statute/case name, short parenthetical).
- Add jurisdiction-specific note if relevant.
- Use plain language, avoid speculation.
- Keep the entire response under 120 words.

Question: {question}
Context: {context}
"""

def get_chat_model() -> ChatGroq:
    """Initialize the Groq chat model"""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        temperature=0.3,
        model_name="llama3-8b-8192"
    )

def create_legal_chain() -> LLMChain:
    """Create a chain with the legal prompt template"""
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=LEGAL_SYSTEM_PROMPT
    )
    model = get_chat_model()
    return LLMChain(llm=model, prompt=prompt)

def get_legal_response(question: str, context: str = "") -> Dict[str, Any]:
    """Get response for legal questions"""
    if not question:
        return {"error": "Question cannot be empty"}
    
    try:
        chain = create_legal_chain()
        response = chain.run(
            question=question,
            context=context if context else "No additional context provided"
        )
        return {"response": response}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def test_chat():
    """Test the legal chat functionality"""
    test_question = "What are the basic requirements for a valid contract in California?"
    result = get_legal_response(test_question, "Business law context")
    print(f"Question: {test_question}")
    print(f"Response: {result}")

if __name__ == "__main__":
    test_chat()