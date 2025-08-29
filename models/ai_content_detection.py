
"""
AI-Generated Content Detection using GPT-2 classifier
Returns: dict with ai_content_analysis (score, explanation)
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.error_handling import error_response

GPT2_DETECTOR_MODEL = "roberta-base-openai-detector"

def get_gpt2_pipeline():
    try:
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(GPT2_DETECTOR_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(GPT2_DETECTOR_MODEL)
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    except Exception:
        return None

def detect_ai_content(text):
    if not text or not isinstance(text, str):
        return error_response("Input text missing or invalid for AI content detection.")
    pipe = get_gpt2_pipeline()
    if pipe is None:
        return error_response("GPT-2 classifier not available.")
    try:
        result = pipe(text)
        score = result[0]["score"]
        label = result[0]["label"]
        explanation = f"{'High' if score > 0.8 else 'Low'} likelihood of AI-generated content due to {label} classification and score {score}."
        return {
            "ai_content_analysis": {
                "score": score,
                "explanation": explanation
            }
        }
    except Exception as e:
        return error_response(f"AI content detection error: {str(e)}")
   
