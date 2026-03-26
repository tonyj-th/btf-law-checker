"""
BtF Law Checker — Knowledge Base Builder

Downloads the pythainlp/thailaw-v1.0 dataset from HuggingFace,
parses each act into section-level chunks, embeds via Cohere,
and stores in ChromaDB + a JSON index for exact lookups.

Usage:
    pip install -r requirements-build.txt
    COHERE_API_KEY=xxx python build_kb.py --output ./kb_data
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import cohere
    import chromadb
    from chromadb.config import Settings
except ImportError:
    cohere = None
    chromadb = None
    Settings = None

# ─── Config ──────────────────────────────────────────────────────────────────

COHERE_MODEL = "embed-multilingual-v3.0"
COHERE_BATCH_SIZE = 96  # Cohere API max
COHERE_DIMS = 1024
MAX_CHUNK_CHARS = 1500  # Max chars per chunk for embedding
CHUNK_OVERLAP_CHARS = 100
COLLECTION_NAME = "thai_law_sections"

# Thai section pattern: มาตรา followed by number, optional /N, optional bis/ter
SECTION_RE = re.compile(
    r'(มาตรา\s+\d+(?:/\d+)?(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ))?)',
    re.UNICODE
)

# Amendment detection
AMENDMENT_RE = re.compile(r'\(แก้ไขเพิ่มเติม[^)]*\)', re.UNICODE)
AMENDMENT_YEAR_RE = re.compile(r'พ\.ศ\.\s*(\d{4})', re.UNICODE)


# ─── Section Parser ──────────────────────────────────────────────────────────

def parse_sections(act_title: str, full_text: str) -> list[dict]:
    """Split a full act text into section-level chunks."""
    if not full_text or not full_text.strip():
        return []

    parts = SECTION_RE.split(full_text)
    sections = []

    # parts[0] is preamble (before first มาตรา)
    preamble = parts[0].strip()
    if preamble and len(preamble) > 50:
        sections.append({
            "act_title": act_title,
            "section_number": "preamble",
            "text": preamble[:MAX_CHUNK_CHARS],
            "amendment_info": None,
        })

    # Alternating: section_header, section_body
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Extract section number from header
        num_match = re.search(
            r'(\d+(?:/\d+)?(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ))?)',
            header
        )
        section_num = num_match.group(1).strip() if num_match else header

        full_section = f"{header} {body}"

        # Detect amendment info
        amendment = None
        amend_match = AMENDMENT_RE.search(full_section)
        if amend_match:
            amendment = amend_match.group(0)

        # Detect B.E. year references
        year_match = AMENDMENT_YEAR_RE.search(full_section)
        be_year = year_match.group(1) if year_match else None

        # If section is very long, split into sub-chunks
        if len(full_section) > MAX_CHUNK_CHARS:
            sub_chunks = split_long_text(full_section)
            for j, chunk in enumerate(sub_chunks):
                sections.append({
                    "act_title": act_title,
                    "section_number": f"{section_num}" if j == 0 else f"{section_num}_part{j+1}",
                    "text": chunk,
                    "amendment_info": amendment,
                    "be_year": be_year,
                })
        else:
            sections.append({
                "act_title": act_title,
                "section_number": section_num,
                "text": full_section,
                "amendment_info": amendment,
                "be_year": be_year,
            })

    # If no sections found (e.g., Royal Decrees), store as single chunk
    if not sections and full_text.strip():
        sections.append({
            "act_title": act_title,
            "section_number": "full",
            "text": full_text[:MAX_CHUNK_CHARS * 3],
            "amendment_info": None,
        })

    return sections


def split_long_text(text: str) -> list[str]:
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            # Try to break at sentence boundary
            last_period = text.rfind('。', start + int(MAX_CHUNK_CHARS * 0.7), end)
            last_space = text.rfind(' ', start + int(MAX_CHUNK_CHARS * 0.7), end)
            if last_period > 0:
                end = last_period + 1
            elif last_space > 0:
                end = last_space + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS
    return chunks


# ─── Build Functions ─────────────────────────────────────────────────────────

def download_dataset() -> list[dict]:
    """Download the Thai law dataset from HuggingFace."""
    print("📥 Downloading pythainlp/thailaw-v1.0 from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("pythainlp/thailaw-v1.0", split="train")
        records = [{"title": row["title"], "text": row["text"]} for row in ds]
        print(f"   ✓ Downloaded {len(records)} acts")
        return records
    except Exception as e:
        print(f"   ✗ datasets library failed: {e}")
        print("   Trying pandas fallback...")
        import pandas as pd
        df = pd.read_parquet("hf://datasets/pythainlp/thailaw-v1.0/data/train-00000-of-00001.parquet")
        records = [{"title": row["title"], "text": row["text"]} for _, row in df.iterrows()]
        print(f"   ✓ Downloaded {len(records)} acts via pandas")
        return records


def build_section_index(all_sections: list[dict]) -> dict:
    """Build a nested dict: {act_title: {section_number: text}} for exact lookups."""
    index = {}
    for s in all_sections:
        act = s["act_title"]
        sec = s["section_number"]
        if act not in index:
            index[act] = {}
        # For sub-chunks (e.g., "19_part2"), only keep the first part for exact lookup
        base_section = sec.split("_")[0]
        if base_section not in index[act]:
            index[act][base_section] = s["text"]
        else:
            # Append if this is a continuation
            if "_part" in sec:
                index[act][base_section] += " " + s["text"]
    return index


def embed_and_store(
    all_sections: list[dict],
    output_dir: Path,
    cohere_key: str,
):
    """Embed all sections and store in ChromaDB."""
    co = cohere.ClientV2(api_key=cohere_key)

    # Initialize ChromaDB
    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"🔢 Embedding {len(all_sections)} chunks via Cohere {COHERE_MODEL}...")
    total_batches = (len(all_sections) + COHERE_BATCH_SIZE - 1) // COHERE_BATCH_SIZE
    embedded_count = 0

    for batch_idx in range(0, len(all_sections), COHERE_BATCH_SIZE):
        batch = all_sections[batch_idx:batch_idx + COHERE_BATCH_SIZE]
        batch_num = batch_idx // COHERE_BATCH_SIZE + 1
        texts = [s["text"] for s in batch]

        # Skip empty texts
        texts = [t if t.strip() else "empty" for t in texts]

        try:
            response = co.embed(
                texts=texts,
                model=COHERE_MODEL,
                input_type="search_document",
                embedding_types=["float"],
            )
            embeddings = response.embeddings.float_

            ids = [f"{s['act_title']}__s{s['section_number']}__{batch_idx + j}"
                   for j, s in enumerate(batch)]
            metadatas = [{
                "act_title": s["act_title"],
                "section_number": s["section_number"],
                "amendment_info": s.get("amendment_info") or "",
                "be_year": s.get("be_year") or "",
            } for s in batch]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            embedded_count += len(batch)
            if batch_num % 50 == 0 or batch_num == total_batches:
                print(f"   Batch {batch_num}/{total_batches} — {embedded_count} chunks embedded")

        except Exception as e:
            print(f"   ✗ Batch {batch_num} failed: {e}")
            # Rate limit handling
            if "rate" in str(e).lower() or "429" in str(e):
                print("   Waiting 60s for rate limit...")
                time.sleep(60)
                # Retry
                try:
                    response = co.embed(
                        texts=texts,
                        model=COHERE_MODEL,
                        input_type="search_document",
                        embedding_types=["float"],
                    )
                    embeddings = response.embeddings.float_
                    ids = [f"{s['act_title']}__s{s['section_number']}__{batch_idx + j}"
                           for j, s in enumerate(batch)]
                    metadatas = [{
                        "act_title": s["act_title"],
                        "section_number": s["section_number"],
                        "amendment_info": s.get("amendment_info") or "",
                        "be_year": s.get("be_year") or "",
                    } for s in batch]
                    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
                    embedded_count += len(batch)
                except Exception as e2:
                    print(f"   ✗ Retry failed: {e2}")
            continue

        # Small delay to avoid rate limits
        if batch_num < total_batches:
            time.sleep(0.3)

    print(f"   ✓ Embedded {embedded_count} chunks into ChromaDB")
    return collection


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build the Thai Law Knowledge Base")
    parser.add_argument("--output", default="./kb_data", help="Output directory")
    parser.add_argument("--skip-embed", action="store_true", help="Only build JSON index, skip embeddings")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of acts to process (0=all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_key and not args.skip_embed:
        print("✗ COHERE_API_KEY not set. Use --skip-embed to only build JSON index.")
        sys.exit(1)

    # Step 1: Download dataset
    records = download_dataset()
    if args.limit > 0:
        records = records[:args.limit]
        print(f"   (Limited to {args.limit} acts)")

    # Step 2: Parse into sections
    print(f"📋 Parsing {len(records)} acts into sections...")
    all_sections = []
    acts_with_sections = 0

    for i, record in enumerate(records):
        sections = parse_sections(record["title"], record["text"])
        if sections:
            all_sections.extend(sections)
            acts_with_sections += 1
        if (i + 1) % 5000 == 0:
            print(f"   Parsed {i+1}/{len(records)} acts → {len(all_sections)} sections so far")

    print(f"   ✓ {len(all_sections)} total sections from {acts_with_sections} acts")

    # Step 3: Build JSON index
    print("📑 Building section index...")
    section_index = build_section_index(all_sections)
    index_path = output_dir / "section_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(section_index, f, ensure_ascii=False, indent=None)
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Index saved: {len(section_index)} acts, {index_size_mb:.1f} MB")

    # Step 4: Save build metadata
    meta = {
        "source": "pythainlp/thailaw-v1.0",
        "total_acts": len(records),
        "acts_with_sections": acts_with_sections,
        "total_sections": len(all_sections),
        "build_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "cohere_model": COHERE_MODEL if not args.skip_embed else "none",
        "collection_name": COLLECTION_NAME,
    }
    with open(output_dir / "kb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Step 5: Embed and store in ChromaDB
    if not args.skip_embed:
        embed_and_store(all_sections, output_dir, cohere_key)
    else:
        print("⏭  Skipping embeddings (--skip-embed)")

    print(f"\n✅ Knowledge base built at {output_dir}/")
    print(f"   - section_index.json ({index_size_mb:.1f} MB)")
    print(f"   - kb_meta.json")
    if not args.skip_embed:
        print(f"   - chroma/ (vector database)")


if __name__ == "__main__":
    main()
