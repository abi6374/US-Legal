"""
Contract Clause Detection using LegalBERT/DeBERTa (CUAD)
Returns: dict with summary, metadata, and clauses (name, text, risk, classification)
"""
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from utils.metadata_extraction import extract_metadata
from utils.error_handling import error_response
import re
from typing import List, Dict, Any

LEGALBERT_MODEL = "mauro/bert-base-uncased-finetuned-clause-type"

def get_clause_pipeline():
    try:
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(LEGALBERT_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(LEGALBERT_MODEL)
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device
        )
    except Exception as e:
        return None

def classify_risk(clause_type: str) -> str:
    high_risk = {"Termination", "Indemnification", "Limitation of Liability"}
    medium_risk = {"Confidentiality", "Non-Compete", "Governing Law"}
    if clause_type in high_risk:
        return "High"
    if clause_type in medium_risk:
        return "Medium"
    return "Low"

def _merge_spans(ents: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    if not ents:
        return []
    ents = sorted(ents, key=lambda e: e.get("start", 0))
    merged = []
    cur = dict(ents[0])
    for e in ents[1:]:
        same_group = e.get("entity_group") == cur.get("entity_group")
        contiguous = e.get("start", 0) <= cur.get("end", 0) + 1
        if same_group and contiguous:
            cur["end"] = max(cur.get("end", 0), e.get("end", 0))
            cur["score"] = max(cur.get("score", 0.0), e.get("score", 0.0))
        else:
            merged.append(cur)
            cur = dict(e)
    merged.append(cur)

    cleaned = []
    for m in merged:
        s = m.get("start")
        e = m.get("end")
        phrase = text[s:e] if s is not None and e is not None else m.get("word", "")
        phrase = phrase.replace("##", "").strip()
        phrase = re.sub(r"\s+", " ", phrase)

        # Filters: drop punctuation-only, too short, or single-token noise
        if not phrase or not re.search(r"[A-Za-z]", phrase):
            continue
        if len(phrase) < 6:
            continue
        # Require at least two words (cuts “Agreement”, “California”, “terminate”)
        if len(phrase.split()) < 2:
            continue
        # Trim leading/trailing punctuation tokens
        phrase = re.sub(r"^[\.\,\;\:\-\–\—\(\)\[\]\'\"\s]+|[\.\,\;\:\-\–\—\(\)\[\]\'\"\s]+$", "", phrase)
        # Drop phrases that start with lowercase stopwords or stray connectors
        if re.match(r"^(and|or|but|the|a|an|of|to|for|in|on|at|by|with)\b", phrase.strip().lower()):
            # keep if has 3+ words and length is meaningful
            if len(phrase.split()) < 3:
                continue

        cleaned.append({
            "entity_group": m.get("entity_group", "Unknown"),
            "text": phrase,
            "start": s,
            "end": e,
            "score": m.get("score", 0.0)
        })
    return cleaned

def _split_sentences(text: str) -> List[Dict[str, int]]:
    # Simple sentence splitter on punctuation; keeps char ranges
    bounds = []
    start = 0
    for m in re.finditer(r"[\.!\?]\s+|\n+", text):
        end = m.start()
        if end > start:
            bounds.append({"start": start, "end": end})
        start = m.end()
    if start < len(text):
        bounds.append({"start": start, "end": len(text)})
    return bounds

def _combine_spans_within_sentences(spans: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    if not spans:
        return []
    sentences = _split_sentences(text)
    combined: List[Dict[str, Any]] = []

    for sent in sentences:
        s0, e0 = sent["start"], sent["end"]
        # take spans that intersect this sentence
        in_sent = [sp for sp in spans if sp.get("start", 0) < e0 and sp.get("end", 0) > s0]
        if not in_sent:
            continue
        # group by entity_group
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for sp in in_sent:
            k = sp.get("entity_group", "Unknown")
            groups.setdefault(k, []).append(sp)
        for k, arr in groups.items():
            # compute min/max offsets of this type within the sentence
            min_s = max(min(sp.get("start", e0) for sp in arr), s0)
            max_e = min(max(sp.get("end", s0) for sp in arr), e0)
            phrase = text[min_s:max_e].strip()
            phrase = re.sub(r"\s+", " ", phrase)
            phrase = re.sub(r"^[\.\,\;\:\-\–\—\(\)\[\]\'\"\s]+|[\.\,\;\:\-\–\—\(\)\[\]\'\"\s]+$", "", phrase)

            # Filter short/noisy combinations
            if not phrase or len(phrase) < 12 or len(phrase.split()) < 3:
                continue

            combined.append({
                "entity_group": k,
                "text": phrase,
                "start": min_s,
                "end": max_e
            })

    # Dedup by (group, text)
    seen = set()
    uniq = []
    for sp in combined:
        key = (sp["entity_group"], sp["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)
    return uniq

def _normalize_group(name: str) -> str:
    if not name:
        return "Unknown"
    mapping = {
        "Governing_Law": "Governing Law",
        "Revenue/Profit_Sharing": "Revenue/Profit Sharing",
        "Termination_Services": "Termination",
        "Effective_Date": "Effective Date",
        "Expiration_Date": "Expiration Date",
        "License_Grant": "License Grant",
        "Audit_Rights": "Audit Rights",
        "Covenant_Not_To_Sue": "Covenant Not To Sue",
        "Transferable_License": "License",
        "Parties": "Parties",
        "Confidentiality": "Confidentiality",
        "Termination": "Termination",
        "Governing Law": "Governing Law",
    }
    return mapping.get(name, name.replace("_", " "))

def _containing_sentence(text: str, s: int, e: int) -> str:
    start = max(text.rfind('.', 0, s), text.rfind('?', 0, s), text.rfind('!', 0, s))
    end_candidates = [text.find('.', e), text.find('?', e), text.find('!', e)]
    end_candidates = [x for x in end_candidates if x != -1]
    end = min(end_candidates) if end_candidates else len(text) - 1
    sent = text[start + 1:end + 1].strip()
    return re.sub(r"\s+", " ", sent)

def _refine_phrase(name: str, phrase: str) -> str:
    if name == "Governing Law":
        m = re.search(r"(governed by the laws? of [A-Za-z ,]+)", phrase, re.I)
        if m:
            return m.group(1).strip().rstrip(".")
    return phrase.strip().rstrip(".")

def detect_clauses(text):
    if not text or not isinstance(text, str):
        return error_response("Input text missing or invalid for clause detection.")
    pipe = get_clause_pipeline()
    if pipe is None:
        return error_response("LegalBERT/DeBERTa model not available.")
    try:
        ner_results = pipe(text)  # returns start/end offsets
        spans = _merge_spans(ner_results, text)
        spans = _combine_spans_within_sentences(spans, text)

        clauses = []
        for sp in spans:
            raw = sp.get("entity_group", "Unknown")
            name = _normalize_group(raw)
            risk = classify_risk(name)

            # Expand fragment → sentence when short
            phrase = sp.get("text", "").strip()
            if len(phrase.split()) < 4 or len(phrase) < 15:
                s = sp.get("start")
                e = sp.get("end")
                if s is not None and e is not None:
                    sent = _containing_sentence(text, s, e)
                    if len(sent.split()) >= 4:
                        phrase = sent

            # Final refinement by clause type
            phrase = _refine_phrase(name, phrase)

            clauses.append({
                "name": name.replace(" ", "_"),
                "text": phrase,
                "risk": risk,
                "classification": name.replace(" ", "_")
            })

        # Deduplicate
        seen = set()
        unique_clauses = []
        for c in clauses:
            k = (c["name"], c["text"].lower())
            if k in seen:
                continue
            seen.add(k)
            unique_clauses.append(c)

        metadata = extract_metadata(text)
        bullets = [
            f"- {c['name'].replace('_', ' ')}: {c['text']}"
            for c in unique_clauses
        ]
        markdown = "\n".join(bullets)
        html = "<ul>" + "".join(
            f"<li>{c['name'].replace('_',' ')}: {c['text']}</li>"
            for c in unique_clauses
        ) + "</ul>"

        return {
            "summary": "Contract analyzed for clauses and risks.",
            "metadata": metadata,
            "clauses": unique_clauses,
            "formatted": markdown,   # keeps backward compatibility
            "bullets": bullets,      # array of bullet strings
            "html": html             # ready-to-render HTML list
        }
    except Exception as e:
        return error_response(f"Clause detection error: {str(e)}")