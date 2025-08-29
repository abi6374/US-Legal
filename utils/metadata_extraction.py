
import re
from .error_handling import error_response

def extract_metadata(text):
    """
    Extracts parties, effective date, and jurisdiction from contract text.
    Returns: dict with parties, effective_date (ISO), jurisdiction
    """
    if not text or not isinstance(text, str):
        return error_response("Input text missing or invalid for metadata extraction.")
    try:
        parties = re.findall(r"Party\s+[A-Z][a-zA-Z]*", text)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        jurisdiction_match = re.search(r"jurisdiction:?\s*([\w ]+)", text, re.IGNORECASE)
        return {
            "parties": parties if parties else ["Party A", "Party B"],
            "effective_date": date_match.group(1) if date_match else None,
            "jurisdiction": jurisdiction_match.group(1).strip() if jurisdiction_match else "US"
        }
    except Exception as e:
        return error_response(f"Metadata extraction error: {str(e)}")
