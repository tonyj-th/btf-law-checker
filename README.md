# BtF Thai Law Citation Checker

A web-based tool that extracts and verifies Thai legal citations from BtF documents against authoritative sources.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run the server
```bash
uvicorn app:app --reload --port 8000
```

### 4. Open in browser
```
http://localhost:8000
```

## How It Works

1. **Upload** a DOCX, PDF, or text file (or paste text directly)
2. **Extraction** — Claude analyses the full document (chunked if needed) and extracts every Thai legal citation
3. **Verification** — each citation is verified via Claude + web search against authoritative sources (krisdika.go.th, Thailand Law Online, Thai Law Forum, etc.)
4. **Report** — colour-coded results with confidence scores, source text, issue details, and suggested corrections

## Architecture

- **Backend**: FastAPI (Python) — handles file parsing, Claude API calls, and SSE streaming
- **Frontend**: Vanilla HTML/CSS/JS — no build step required
- **Progress**: Server-Sent Events for real-time updates
- **File parsing**: mammoth (DOCX), native PDF support via Claude, plain text

## Deployment

For production deployment (e.g., on Railway, Render, or a VPS):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Environment Variables
- `ANTHROPIC_API_KEY` — required, your Anthropic API key

## Files

```
btf-law-checker/
├── app.py              # FastAPI backend
├── static/
│   └── index.html      # Frontend (self-contained)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Notes

- The tool uses Claude Sonnet with web search for verification — each citation takes ~10-15 seconds
- Documents are chunked at ~10K characters with 500-char overlap to avoid missing citations at boundaries
- Job results are stored in memory (swap for Redis/DB in production)
- Jobs auto-expire after 2 hours
