# models/courtlistener.py
import os, re, requests
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://www.courtlistener.com/api/rest/v4/search/"
API_KEY = os.getenv("COURTLISTENER_API_KEY", "").strip()

def get_case_from_courtlistener(citation_text: str) -> Optional[Dict[str, Any]]:
    """
    Look up a legal opinion/statute using CourtListener's /search/.
    Returns a dict with case metadata or None if not found.
    """
    if not citation_text or not isinstance(citation_text, str):
        return None

    params: Dict[str, Any] = {}
    # Case like: 410 U.S. 113 (1973)
    case_match = re.search(r'(\d+\s+U\.S\.\s+\d+)\s*\((\d{4})\)', citation_text)
    # Statute like: 42 U.S.C. § 1983
    statute_match = re.search(r'(\d+\s+U\.S\.C\.\s+§?\s*\d+)', citation_text)

    if case_match:
        reporter_citation = case_match.group(1)
        year = case_match.group(2)
        params["q"] = f'"{reporter_citation}" AND {year}'
        params["type"] = "o"  # opinions
    elif statute_match:
        params["q"] = statute_match.group(1)
    else:
        params["q"] = f'"{citation_text}"'

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Token {API_KEY}"

    try:
        resp = requests.get(API_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("results"):
            return None

        case = data["results"][0]

        # Court name resolution
        court_obj = case.get("court")
        court_name = "Unknown"
        if isinstance(court_obj, str):
            if court_obj.startswith("http"):
                try:
                    court_resp = requests.get(court_obj, headers=headers, timeout=5)
                    if court_resp.status_code == 200:
                        cj = court_resp.json()
                        court_name = cj.get("full_name", cj.get("short_name", "Unknown"))
                    else:
                        court_name = court_obj.strip("/").split("/")[-1].upper()
                except Exception:
                    court_name = court_obj.strip("/").split("/")[-1].upper()
            else:
                court_name = court_obj.upper()
        elif isinstance(court_obj, dict):
            court_name = court_obj.get("name", "Unknown")

        filed_date_str = case.get("dateFiled")
        year_val = None
        if filed_date_str:
            try:
                year_val = datetime.strptime(filed_date_str, "%Y-%m-%d").year
            except ValueError:
                year_val = None

        case_name = case.get("caseName") or citation_text.split(",")[0]

        return {
            "case_name": case_name or "N/A",
            "court": court_name,
            "date": filed_date_str or "N/A",
            "year": year_val,
            "summary": case.get("snippet", "") or "",
            "plain_text": case.get("plain_text", "") or "",
            "url": case.get("absolute_url", "") or "",
            "query": params.get("q", ""),
        }
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.HTTPError:
        return None
    except requests.exceptions.RequestException:
        return None