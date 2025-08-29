
"""
Legal Citation Extraction & Verification using spaCy NER and CourtListener API
Returns: dict with summary and citations (citation, valid, source_url)
"""
import spacy
import requests
from utils.error_handling import error_response

NER_MODEL = "en_core_web_trf"

def get_spacy_nlp():
    try:
        return spacy.load(NER_MODEL)
    except Exception:
        return None

def extract_and_verify_citations(text):
    if not text or not isinstance(text, str):
        return error_response("Input text missing or invalid for citation verification.")
    nlp = get_spacy_nlp()
    if nlp is None:
        return error_response("spaCy transformer NER not available.")
    try:
        doc = nlp(text)
        citations = []
        for ent in doc.ents:
            if ent.label_ in ["LAW", "CASE"]:
                citation = ent.text
                url = f"https://www.courtlistener.com/api/rest/v3/search/?q={citation}"
                try:
                    r = requests.get(url)
                    data = r.json()
                    valid = bool(data.get("results"))
                    source_url = data["results"][0]["absolute_url"] if valid else ""
                    if source_url:
                        source_url = f"https://www.courtlistener.com{source_url}"
                except Exception:
                    valid = False
                    source_url = ""
                citations.append({
                    "citation": citation,
                    "valid": valid,
                    "source_url": source_url
                })
        return {
            "summary": "Legal citations extracted and verified.",
            "citations": citations
        }
    except Exception as e:
        return error_response(f"Citation verification error: {str(e)}")
