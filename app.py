"""
BtF Thai Law Citation Checker — FastAPI Backend
Handles document upload, citation extraction via Claude, and verification via Claude + web search.
Progress is streamed to the frontend via Server-Sent Events (SSE).
"""

import os
import json
import uuid
import asyncio
import time
import re
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

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6-20250514"
CHUNK_SIZE = 10_000
CHUNK_OVERLAP = 500
MAX_RETRIES = 2
TIMEOUT_SECONDS = 60
CACHE_TTL_DAYS = 30
DB_PATH = Path(__file__).parent / "citation_cache.db"

# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB + launch background cleanup task
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

app = FastAPI(title="BtF Law Checker", version="0.3.0", lifespan=lifespan)
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_client():
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


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


async def verify_single_citation(citation: dict, index: int, total: int, job_id: str, db: aiosqlite.Connection):
    """Verify a single citation using Claude with web search tool, with caching."""
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

    # ─── Cache miss — call Claude ─────────────────────────────────────
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
                job["log"].append(f"  → {parsed['status']} ({parsed.get('confidence', '?')}%)")
                # Store in cache
                await cache_set(db, cache_key, citation, parsed)
                return parsed
            else:
                job["log"].append(f"  → Could not parse verification response")
                return _unverifiable("Could not parse verification response")

        except Exception as e:
            if attempt < MAX_RETRIES:
                job["log"].append(f"  → Retry {attempt+1}: {str(e)[:80]}")
                await asyncio.sleep(2)
            else:
                job["log"].append(f"  → Error: {str(e)[:80]}")
                return _unverifiable(f"Verification error: {str(e)[:80]}")


def _unverifiable(reason: str) -> dict:
    return {
        "status": "unverifiable",
        "confidence": 0,
        "source_text": "",
        "issue_details": reason,
        "suggested_correction": "",
        "amendment_status": "",
        "sources_checked": []
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

        for i, citation in enumerate(citations):
            if job.get("cancelled"):
                job["log"].append("Job cancelled by user")
                break

            result = await verify_single_citation(citation, i, total, job_id, db)
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

