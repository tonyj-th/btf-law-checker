"""
BtF Thai Law Citation Checker — FastAPI Backend
Three-stage verification: Local KB → Semantic RAG → Web Search fallback.
Progress is streamed to the frontend via Server-Sent Events (SSE).
"""

import os
import sys
import json
import uuid
import asyncio
import time
import re
import logging
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

import aiosqlite
import anthropic
import mammoth
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from kb import LawKB

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
CHUNK_SIZE = 10_000
CHUNK_OVERLAP = 500
MAX_RETRIES = 2
TIMEOUT_SECONDS = 60
CACHE_TTL_DAYS = 30
DB_PATH = Path(os.environ.get("DB_DIR", "/tmp")) / "citation_cache.db"

# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB + KB + launch background cleanup task
    db = await aiosqlite.connect(str(DB_PATH))
    await db.execute("""
        CREATE TABLE IF NOT EXISTS citation_cache (
            cache_key TEXT PRIMARY KEY,
            citation_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    await db.commit()
    app.state.db = db

    # Initialize Knowledge Base (tries volume first, then bundled priority index)
    kb_dir = os.environ.get("KB_DIR", "/data/kb")
    try:
        app.state.kb = LawKB(kb_dir)
        if app.state.kb.is_available():
            logger.info(f"KB loaded: {app.state.kb.act_count} acts, {app.state.kb.section_count} sections")
        else:
            logger.info("KB not available — running in web-search-only mode")
            app.state.kb = None
    except Exception as e:
        logger.warning(f"KB failed to load: {e}")
        app.state.kb = None

    async def _cleanup():
        while True:
            await asyncio.sleep(3600)
            cutoff = time.time() - 7200  # 2 hours
            to_remove = [jid for jid, j in jobs.items() if j["created_at"] < cutoff]
            for jid in to_remove:
                del jobs[jid]
    cleanup_task = asyncio.create_task(_cleanup())
    yield
    # Shutdown
    cleanup_task.cancel()
    await db.close()

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="BtF Law Checker", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory job store (swap for Redis in production)
jobs: dict = {}

# ─── Prompts ──────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a Thai legal citation extractor for Better Than Freehold (BtF), a lease securitisation and foreign property ownership platform.

Analyse the following document text and extract EVERY legal citation, reference, or claim about Thai law. Be thorough — catch both explicit citations (e.g., "Section 19 of the Condominium Act") and implicit legal claims (e.g., "foreigners cannot own land in Thailand").

For each citation found, provide:
- citation_text: the exact text from the document containing the reference
- type: one of "statute", "act_reference", "case_law", "constitution", "regulation", "legal_principle", "amendment"
- act_name: the name of the Act or law (if identifiable)
- section: the section number (if specified)
- year_be: the Buddhist Era year (if mentioned)
- year_ad: the AD year equivalent (if determinable)
- context: a brief note on what the document claims about this law

Respond ONLY with a JSON array. No markdown, no backticks, no preamble. If no citations are found, return [].
"""

VERIFICATION_PROMPT = """You are a Thai law verification expert. You have been given a legal citation extracted from a BtF (Better Than Freehold) document.

Your task is to verify whether the citation is accurate. Use the web_search tool to look up the relevant Thai law from authoritative sources such as krisdika.go.th, thailandlawonline.com, thailawforum.com, or library.siam-legal.com.

IMPORTANT: If the citation is inaccurate, outdated, or partially incorrect, you MUST search for and provide the CORRECT legal reference. This includes:
- The correct Act name (with B.E. year)
- The correct section number
- The actual text of the correct provision
- Any relevant Thai Supreme Court decisions (e.g. "Supreme Court Decision No. 1234/2550")
- The source URL where the correct law can be found

After searching, evaluate and respond with ONLY a JSON object (no markdown, no backticks, no preamble):
{
  "status": "verified" | "partially_verified" | "inaccurate" | "outdated" | "unverifiable",
  "confidence": <number 0-100>,
  "source_text": "<actual text from the authoritative source>",
  "issue_details": "<explanation of any discrepancy>",
  "suggested_correction": "<recommended rewrite if inaccurate>",
  "amendment_status": "<whether this provision has been amended>",
  "sources_checked": ["<list of source names checked>"],
  "correct_reference": {
    "act_name": "<correct Act name with B.E. year, or null if verified>",
    "section": "<correct section number, or null if verified>",
    "full_text": "<actual text of the correct provision, or null if verified>",
    "source_url": "<URL of the authoritative source, or null if verified>",
    "case_law": "<relevant Supreme Court decision numbers and brief summary, or null if none>",
    "notes": "<brief explanation of why this is the correct reference, or null if verified>"
  }
}

For "verified" citations, set correct_reference fields to null. For "inaccurate" or "outdated" citations, you MUST populate correct_reference with the actual correct law. For "partially_verified", populate correct_reference if there are specific corrections needed.

Be strict: only mark as "verified" if the search results clearly confirm accuracy. Mark as "partially_verified" or "unverifiable" if results are ambiguous or insufficient.
"""

KB_VERIFICATION_PROMPT = """You are a Thai law verification expert. Compare a legal citation from a BtF (Better Than Freehold) document against the ACTUAL Thai legal text retrieved from the official knowledge base (sourced from krisdika.go.th).

The knowledge base text is in Thai. The citation is in English. You must:
1. Translate/interpret the Thai legal text to understand what it actually says
2. Compare the English citation's claims against the actual Thai law
3. Determine if the citation accurately represents the law

IMPORTANT: If the citation is inaccurate, outdated, or partially incorrect, provide the CORRECT legal reference including:
- The correct Act name (with B.E. year)
- The correct section number
- What the law actually says (translate the relevant Thai text)
- Any relevant Supreme Court decisions if applicable

Respond with ONLY a JSON object (no markdown, no backticks, no preamble):
{
  "status": "verified" | "partially_verified" | "inaccurate" | "outdated" | "unverifiable",
  "confidence": <number 0-100>,
  "source_text": "<translated summary of the actual Thai legal text>",
  "issue_details": "<explanation of any discrepancy>",
  "suggested_correction": "<recommended rewrite if inaccurate>",
  "amendment_status": "<whether this provision has been amended>",
  "sources_checked": ["Official Thai Law KB (krisdika.go.th)"],
  "correct_reference": {
    "act_name": "<correct Act name with B.E. year, or null if verified>",
    "section": "<correct section number, or null if verified>",
    "full_text": "<actual text of the correct provision, or null if verified>",
    "source_url": "<URL or null>",
    "case_law": "<relevant Supreme Court decisions or null>",
    "notes": "<explanation or null>"
  }
}

For "verified" citations, set correct_reference fields to null. Be strict but fair.
"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using mammoth."""
    import io
    result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
    return result.value


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks, breaking at paragraph/sentence boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # Try paragraph boundary first
            last_para = text.rfind('\n\n', start + int(CHUNK_SIZE * 0.7), end)
            last_sentence = text.rfind('. ', start + int(CHUNK_SIZE * 0.7), end)
            if last_para > 0:
                end = last_para + 2
            elif last_sentence > 0:
                end = last_sentence + 2
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_search_query(citation: dict) -> str:
    parts = []
    if citation.get("act_name"):
        parts.append(f'Thailand "{citation["act_name"]}"')
    if citation.get("section"):
        parts.append(f'Section {citation["section"]}')
    if citation.get("year_be"):
        parts.append(f'B.E. {citation["year_be"]}')
    if not parts:
        parts.append(f'Thailand law {citation.get("citation_text", "")[:50]}')
    return " ".join(parts)


def parse_json_from_response(text: str) -> Optional[dict | list]:
    """Try to extract JSON from a Claude response that might contain extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    cleaned = re.sub(r'```json\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON array or object
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*?"status"[\s\S]*?\}']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


# ─── Cache ────────────────────────────────────────────────────────────────────

# Common abbreviation mappings for fuzzy cache key matching
_ACT_ALIASES = {
    "condo act": "condominium act",
    "condominium act": "condominium act",
    "land code": "land code",
    "fba": "foreign business act",
    "foreign business act": "foreign business act",
    "ccc": "civil and commercial code",
    "civil and commercial code": "civil and commercial code",
}


def make_cache_key(citation: dict) -> str:
    """Generate a normalized cache key from citation fields."""
    act = (citation.get("act_name") or "").strip().lower()
    # Normalize common abbreviations
    for alias, canonical in _ACT_ALIASES.items():
        if alias in act:
            act = canonical
            break
    section = (str(citation.get("section") or "")).strip().lower()
    ctype = (citation.get("type") or "").strip().lower()
    key = f"{act}|{section}|{ctype}"
    # Only cache if we have meaningful identifiers
    if act or section:
        return key
    return ""


async def cache_get(db: aiosqlite.Connection, key: str) -> Optional[dict]:
    """Look up a cached verification result. Returns None on miss or expiry."""
    if not key:
        return None
    async with db.execute(
        "SELECT result_json FROM citation_cache WHERE cache_key = ? AND expires_at > ?",
        (key, time.time()),
    ) as cursor:
        row = await cursor.fetchone()
        if row:
            result = json.loads(row[0])
            result["_cached"] = True
            return result
    return None


async def cache_set(db: aiosqlite.Connection, key: str, citation: dict, result: dict):
    """Store a verification result in the cache."""
    if not key:
        return
    now = time.time()
    await db.execute(
        "INSERT OR REPLACE INTO citation_cache (cache_key, citation_json, result_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
        (key, json.dumps(citation), json.dumps(result), now, now + CACHE_TTL_DAYS * 86400),
    )
    await db.commit()


# ─── Core Logic ───────────────────────────────────────────────────────────────

async def extract_citations_from_text(text: str, job_id: str):
    """Extract all legal citations from document text using Claude, with chunking."""
    client = get_client()
    chunks = chunk_text(text)
    all_citations = []
    job = jobs[job_id]
    
    job["log"].append(f"Document: {len(text)} chars, split into {len(chunks)} chunk(s)")
    job["stage"] = "extracting"
    
    for i, chunk in enumerate(chunks):
        job["log"].append(f"Extracting from chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        job["extraction_progress"] = {"current": i, "total": len(chunks)}
        
        chunk_label = f"[CHUNK {i+1} OF {len(chunks)}]\n\n" if len(chunks) > 1 else ""
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": EXTRACTION_PROMPT + "\n\nDOCUMENT TEXT:\n" + chunk_label + chunk
                    }],
                    timeout=TIMEOUT_SECONDS,
                )
                
                response_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                
                parsed = parse_json_from_response(response_text)
                if isinstance(parsed, list):
                    job["log"].append(f"  → Found {len(parsed)} citation(s) in chunk {i+1}")
                    all_citations.extend(parsed)
                else:
                    job["log"].append(f"  → Warning: could not parse chunk {i+1} response")
                break
                
            except Exception as e:
                if attempt < MAX_RETRIES:
                    job["log"].append(f"  → Retry {attempt+1}: {str(e)[:80]}")
                    await asyncio.sleep(2)
                else:
                    job["log"].append(f"  → Failed chunk {i+1}: {str(e)[:80]}")
    
    # Deduplicate
    seen = set()
    unique = []
    for c in all_citations:
        key = c.get("citation_text", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    
    if len(chunks) > 1:
        job["log"].append(f"Deduplicated: {len(all_citations)} → {len(unique)} unique citations")
    
    return unique


async def extract_citations_from_pdf(pdf_bytes: bytes, job_id: str):
    """Extract citations from PDF by sending it directly to Claude as a document."""
    import base64
    client = get_client()
    job = jobs[job_id]
    
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    job["log"].append(f"Sending PDF ({len(pdf_bytes)} bytes) to Claude...")
    job["stage"] = "extracting"
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": b64}
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT + "\n\n[See the uploaded PDF document above]"
                        }
                    ]
                }],
                timeout=TIMEOUT_SECONDS * 2,  # PDFs can be slower
            )
            
            response_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            
            parsed = parse_json_from_response(response_text)
            if isinstance(parsed, list):
                job["log"].append(f"Found {len(parsed)} citation(s) from PDF")
                return parsed
            else:
                job["log"].append("Warning: could not parse PDF extraction response")
                return []
                
        except Exception as e:
            if attempt < MAX_RETRIES:
                job["log"].append(f"Retry {attempt+1}: {str(e)[:80]}")
                await asyncio.sleep(3)
            else:
                job["log"].append(f"PDF extraction failed: {str(e)[:80]}")
                return []


async def verify_single_citation(citation: dict, index: int, total: int, job_id: str, db: aiosqlite.Connection, kb: Optional["LawKB"] = None):
    """Verify a citation using three-stage pipeline: Local KB → Semantic RAG → Web Search."""
    job = jobs[job_id]

    label = citation.get("act_name", citation.get("citation_text", "")[:60])
    if citation.get("section"):
        label += f" § {citation['section']}"

    job["log"].append(f"Verifying ({index+1}/{total}): {label}")
    job["verification_progress"] = {"current": index, "total": total, "label": label}

    # ─── Check cache first ────────────────────────────────────────────
    cache_key = make_cache_key(citation)
    cached = await cache_get(db, cache_key)
    if cached:
        job["log"].append(f"  → cached: {cached['status']} ({cached.get('confidence', '?')}%)")
        return cached

    # ─── Stage 1: Exact Match (Local KB) ──────────────────────────────
    if kb and kb.is_available() and citation.get("act_name") and citation.get("section"):
        exact = kb.exact_lookup(citation["act_name"], str(citation["section"]))
        if exact:
            match_type = exact.get("match_type", "exact")
            kb_text = exact.get("english_text") or exact.get("thai_text", "")
            job["log"].append(f"  → Stage 1: {match_type} match found ({exact.get('act_name_en', exact.get('act_name_thai', ''))})")
            result = await _verify_against_kb(citation, kb_text, exact.get("act_name_en") or exact.get("act_name_thai", ""), exact["section"])
            if result and result.get("confidence", 0) >= 60:
                result["verification_tier"] = "local_kb"
                if match_type == "primary_statute":
                    result["verification_tier"] = "primary_statute"
                    result["source_urls"] = exact.get("source_urls", [])
                    result["amendment_notes_kb"] = exact.get("amendment_notes", "")
                await cache_set(db, cache_key, citation, result)
                job["log"].append(f"  → {result['status']} ({result.get('confidence', '?')}%) [Local KB - {match_type}]")
                return result
            job["log"].append(f"  → Stage 1: Low confidence ({result.get('confidence', 0)}%), trying next stage")

    # ─── Stage 2: Semantic RAG (ChromaDB + Cohere) ────────────────────
    if kb and kb.has_semantic_search():
        query = f"{citation.get('act_name', '')} {citation.get('citation_text', '')}"
        chunks = await kb.semantic_search(query, n_results=5)
        if chunks and chunks[0]["distance"] < 0.5:
            job["log"].append(f"  → Stage 2: Semantic match (distance={chunks[0]['distance']:.3f})")
            # Build context from top chunks
            kb_context = "\n\n---\n\n".join(
                f"[{c['act_title']} - Section {c['section_number']}]\n{c['text']}"
                for c in chunks[:3]
            )
            result = await _verify_against_kb(citation, kb_context, chunks[0]["act_title"], chunks[0]["section_number"])
            if result and result.get("confidence", 0) >= 50:
                result["verification_tier"] = "semantic_match"
                await cache_set(db, cache_key, citation, result)
                job["log"].append(f"  → {result['status']} ({result.get('confidence', '?')}%) [Semantic]")
                return result
            job["log"].append(f"  → Stage 2: Low confidence, falling back to web search")

    # ─── Stage 3: Web Search Fallback ─────────────────────────────────
    job["log"].append(f"  → Stage 3: Web search")
    result = await _verify_via_web_search(citation, job_id)
    result["verification_tier"] = "web_search"
    await cache_set(db, cache_key, citation, result)
    job["log"].append(f"  → {result['status']} ({result.get('confidence', '?')}%) [Web Search]")
    return result


async def _verify_against_kb(citation: dict, thai_text: str, act_name_thai: str, section: str) -> Optional[dict]:
    """Verify a citation against Thai legal text from the local KB. No web search tool — cheaper."""
    client = get_client()

    prompt = f"""{KB_VERIFICATION_PROMPT}

Citation from BtF document: "{citation.get('citation_text', '')}"
Type: {citation.get('type', 'unspecified')}
Act (English): {citation.get('act_name', 'unspecified')}
Section: {citation.get('section', 'unspecified')}
Year (B.E.): {citation.get('year_be', 'unspecified')}
Context: {citation.get('context', 'none')}

Actual Thai legal text from knowledge base:
Act: {act_name_thai}
Section: {section}
Text:
{thai_text[:3000]}"""

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                timeout=TIMEOUT_SECONDS,
            )

            response_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            parsed = parse_json_from_response(response_text)
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
            return None

        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)
            else:
                logger.error(f"KB verification error: {e}")
                return None


async def _verify_via_web_search(citation: dict, job_id: str) -> dict:
    """Stage 3: Verify using Claude + web search (existing logic)."""
    client = get_client()
    search_query = build_search_query(citation)

    prompt = f"""{VERIFICATION_PROMPT}

Citation: "{citation.get('citation_text', '')}"
Type: {citation.get('type', 'unspecified')}
Act: {citation.get('act_name', 'unspecified')}
Section: {citation.get('section', 'unspecified')}
Year (B.E.): {citation.get('year_be', 'unspecified')}
Context: {citation.get('context', 'none')}

Search for: {search_query}"""

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                timeout=TIMEOUT_SECONDS,
            )

            response_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            parsed = parse_json_from_response(response_text)
            if isinstance(parsed, dict) and "status" in parsed:
                return parsed
            else:
                return _unverifiable("Could not parse verification response")

        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)
            else:
                return _unverifiable(f"Verification error: {str(e)[:80]}")


def _unverifiable(reason: str, tier: str = "web_search") -> dict:
    return {
        "status": "unverifiable",
        "confidence": 0,
        "source_text": "",
        "issue_details": reason,
        "suggested_correction": "",
        "amendment_status": "",
        "sources_checked": [],
        "verification_tier": tier,
    }


async def run_job(job_id: str, file_bytes: bytes, filename: str):
    """Main job runner — extract then verify."""
    job = jobs[job_id]
    
    try:
        # ─── Step 1: Extract text ─────────────────────────────────────────
        job["stage"] = "parsing"
        job["log"].append(f"Processing file: {filename}")
        
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        if ext == "pdf":
            citations = await extract_citations_from_pdf(file_bytes, job_id)
        elif ext == "docx":
            job["log"].append("Extracting text from DOCX...")
            text = extract_text_from_docx(file_bytes)
            job["log"].append(f"Extracted {len(text)} characters")
            citations = await extract_citations_from_text(text, job_id)
        elif ext in ("txt", "md"):
            text = extract_text_from_txt(file_bytes)
            job["log"].append(f"Text file: {len(text)} characters")
            citations = await extract_citations_from_text(text, job_id)
        else:
            job["log"].append(f"Unsupported file type: .{ext}")
            job["stage"] = "error"
            job["error"] = f"Unsupported file type: .{ext}"
            return
        
        job["citations"] = citations
        job["log"].append(f"Extraction complete: {len(citations)} citation(s)")
        
        if not citations:
            job["stage"] = "complete"
            return
        
        # ─── Step 2: Verify each citation ─────────────────────────────────
        job["stage"] = "verifying"
        job["results"] = {}
        total = len(citations)
        db = app.state.db
        kb = getattr(app.state, "kb", None)

        if kb and kb.is_available():
            job["log"].append(f"Knowledge base active: {kb.act_count} acts, {kb.section_count} sections")
        else:
            job["log"].append("Knowledge base not available — using web search only")

        for i, citation in enumerate(citations):
            if job.get("cancelled"):
                job["log"].append("Job cancelled by user")
                break

            result = await verify_single_citation(citation, i, total, job_id, db, kb)
            job["results"][str(i)] = result

            # Small delay to avoid rate limiting (skip for cached results)
            if i < total - 1 and not result.get("_cached"):
                await asyncio.sleep(0.5)
        
        job["stage"] = "complete"
        job["log"].append(f"Verification complete: {len(job['results'])}/{total} citations checked")
        
    except Exception as e:
        job["stage"] = "error"
        job["error"] = str(e)
        job["log"].append(f"Fatal error: {str(e)}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/kb-status")
async def kb_status():
    """Return knowledge base status."""
    kb = getattr(app.state, "kb", None)
    if kb and kb.is_available():
        return kb.status()
    return {"available": False, "semantic_search": False, "act_count": 0, "section_count": 0, "source": "none", "build_date": "N/A", "vector_count": 0}


@app.get("/api/kb-debug-lookup")
async def kb_debug_lookup(act: str = "Condominium Act", section: str = "19"):
    """Debug: trace KB lookup step by step."""
    kb = getattr(app.state, "kb", None)
    if not kb or not kb.is_available():
        return {"error": "KB not available", "available": False}

    # Step 1: resolve act name
    thai_name = kb._resolve_act_name(act)

    # Step 2: check if thai name exists in index
    matched_act = None
    index_keys_sample = list(kb._section_index.keys())[:10]
    if thai_name:
        for index_key in kb._section_index:
            if thai_name in index_key:
                matched_act = index_key
                break

    # Step 3: check section
    sections_available = []
    if matched_act:
        sections_available = list(kb._section_index[matched_act].keys())[:20]

    # Step 4: do the actual lookup
    result = kb.exact_lookup(act, section)

    return {
        "input_act": act,
        "input_section": section,
        "thai_name_resolved": thai_name,
        "matched_act_in_index": matched_act,
        "index_keys_sample": index_keys_sample,
        "sections_available": sections_available,
        "mapping_count": len(kb._act_mapping),
        "index_count": len(kb._section_index),
        "lookup_result": result is not None,
        "result_preview": result["thai_text"][:200] if result else None,
    }


@app.post("/api/kb-build")
async def kb_build(background_tasks: BackgroundTasks):
    """Trigger knowledge base build on the server (downloads dataset, parses sections, saves index)."""
    kb_dir = os.environ.get("KB_DIR", "/data/kb")
    index_path = Path(kb_dir) / "section_index.json"

    if index_path.exists():
        return {"status": "already_built", "message": "KB index already exists. Delete /data/kb/section_index.json to rebuild."}

    # Store build log in app state for debugging
    app.state.kb_build_log = "Building..."

    async def _build():
        import subprocess
        try:
            os.makedirs(kb_dir, exist_ok=True)
            result = subprocess.run(
                [sys.executable, "build_kb.py", "--output", kb_dir, "--skip-embed", "--limit", "5000"],
                capture_output=True, text=True, timeout=1800,
                cwd=str(Path(__file__).parent),
            )
            app.state.kb_build_log = f"exit={result.returncode}\nSTDOUT:\n{result.stdout[-1000:]}\nSTDERR:\n{result.stderr[-1000:]}"
            if result.returncode == 0:
                try:
                    app.state.kb = LawKB(kb_dir)
                    app.state.kb_build_log += f"\nKB loaded: {app.state.kb.act_count} acts"
                except Exception as e:
                    app.state.kb_build_log += f"\nKB reload failed: {e}"
            else:
                app.state.kb_build_log += "\nBUILD FAILED"
        except Exception as e:
            app.state.kb_build_log = f"Exception: {e}"

    background_tasks.add_task(_build)
    return {"status": "building", "message": "Knowledge base build started. Check /api/kb-build-log for progress."}


@app.get("/api/kb-build-log")
async def kb_build_log():
    """Return KB build log for debugging."""
    return {"log": getattr(app.state, "kb_build_log", "No build attempted")}


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a document and start extraction + verification."""
    file_bytes = await file.read()
    
    if len(file_bytes) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")
    
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "stage": "queued",
        "citations": [],
        "results": {},
        "log": [],
        "error": None,
        "cancelled": False,
        "created_at": time.time(),
        "extraction_progress": {"current": 0, "total": 0},
        "verification_progress": {"current": 0, "total": 0, "label": ""},
    }
    
    background_tasks.add_task(run_job, job_id, file_bytes, file.filename)
    
    return {"job_id": job_id}


@app.post("/api/upload-text")
async def upload_text(background_tasks: BackgroundTasks, body: dict):
    """Upload raw text for checking."""
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "filename": "pasted-text.txt",
        "stage": "queued",
        "citations": [],
        "results": {},
        "log": [],
        "error": None,
        "cancelled": False,
        "created_at": time.time(),
        "extraction_progress": {"current": 0, "total": 0},
        "verification_progress": {"current": 0, "total": 0, "label": ""},
    }
    
    background_tasks.add_task(run_job, job_id, text.encode("utf-8"), "pasted-text.txt")
    
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    """Get current job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job (finishes current citation then stops)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs[job_id]["cancelled"] = True
    return {"status": "cancelling"}


@app.get("/api/job/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream for real-time progress updates."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_stream():
        last_log_count = 0
        while True:
            if job_id not in jobs:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break
            
            job = jobs[job_id]
            
            # Send new log entries
            current_log_count = len(job["log"])
            if current_log_count > last_log_count:
                for entry in job["log"][last_log_count:]:
                    yield f"data: {json.dumps({'type': 'log', 'message': entry})}\n\n"
                last_log_count = current_log_count
            
            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'stage': job['stage'], 'extraction_progress': job['extraction_progress'], 'verification_progress': job['verification_progress'], 'citation_count': len(job['citations']), 'result_count': len(job['results'])})}\n\n"
            
            if job["stage"] in ("complete", "error"):
                yield f"data: {json.dumps({'type': 'done', 'stage': job['stage']})}\n\n"
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

