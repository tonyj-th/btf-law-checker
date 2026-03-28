"""Embed only the 146 primary statute sections into ChromaDB."""
import json
import os
import sys
from pathlib import Path

import cohere
import chromadb

COHERE_MODEL = "embed-multilingual-v3.0"
DATA_DIR = Path("data/primary_statutes")
OUTPUT_DIR = Path("kb_data/chroma")

def main():
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_key:
        print("Set COHERE_API_KEY"); sys.exit(1)

    co = cohere.Client(cohere_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(OUTPUT_DIR))

    # Delete existing collection if any
    try:
        client.delete_collection("thai_law_sections")
    except Exception:
        pass
    collection = client.get_or_create_collection("thai_law_sections", metadata={"hnsw:space": "cosine"})

    # Load all primary statute sections
    ids, texts, metadatas = [], [], []
    for f in sorted(DATA_DIR.glob("*.json")):
        data = json.load(open(f, encoding="utf-8"))
        act_en = data["act_name_en"]
        act_th = data.get("act_name_th", "")
        year_be = data.get("year_be", "")
        year_ad = data.get("year_ad", "")

        for sec_num, sec_data in data["sections"].items():
            text_en = sec_data.get("text_en", "")
            text_th = sec_data.get("text_th", "")
            # Combine for embedding — search in both languages
            combined = f"{act_en} Section {sec_num}\n{text_en}"
            if text_th:
                combined += f"\n{text_th}"

            doc_id = f"{f.stem}__s{sec_num}"
            ids.append(doc_id)
            texts.append(combined)
            metadatas.append({
                "act_name_en": act_en,
                "act_name_th": act_th,
                "section": sec_num,
                "year_be": year_be,
                "year_ad": str(year_ad),
                "source_file": f.name,
            })

    print(f"Embedding {len(ids)} primary sections...")

    # Embed in batches of 96
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]

        response = co.embed(
            texts=batch_texts,
            model=COHERE_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        embeddings = response.embeddings.float_

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} done")

    print(f"✅ {len(ids)} sections embedded into {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
