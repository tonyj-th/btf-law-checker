"""
BtF Law Checker — Knowledge Base Runtime Module

Provides three-stage legal citation lookup:
  Stage 1: Exact match via JSON section index
  Stage 2: Semantic search via ChromaDB + Cohere embeddings
  Stage 3: (handled in app.py) Web search fallback

Loads at app startup. Gracefully degrades if KB data is missing.
"""

import gzip
import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LawKB:
    """Thai law knowledge base for citation verification."""

    def __init__(self, kb_dir: str):
        self.kb_dir = Path(kb_dir)
        self._section_index: dict = {}
        self._act_mapping: dict = {}
        self._meta: dict = {}
        self._chroma_collection = None
        self._cohere_client = None
        self._available = False

        self._load()

    def _load(self):
        """Load KB data from disk."""
        # Load section index — try full index first, then bundled priority index
        index_path = self.kb_dir / "section_index.json"
        bundled_path = Path(__file__).parent / "data" / "priority_index.json.gz"

        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._section_index = json.load(f)
                logger.info(f"KB: Loaded FULL section index with {len(self._section_index)} acts")
            except Exception as e:
                logger.error(f"KB: Failed to load section index: {e}")

        if not self._section_index and bundled_path.exists():
            try:
                with gzip.open(bundled_path, "rb") as f:
                    self._section_index = json.loads(f.read().decode("utf-8"))
                self._meta = {"source": "pythainlp/thailaw-v1.0 (priority subset)", "build_date": "2026-03-26", "total_acts": len(self._section_index)}
                logger.info(f"KB: Loaded BUNDLED priority index with {len(self._section_index)} acts")
            except Exception as e:
                logger.error(f"KB: Failed to load bundled index: {e}")
                return

        # Load metadata
        meta_path = self.kb_dir / "kb_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self._meta = json.load(f)

        # Load act mapping (English → Thai)
        mapping_path = Path(__file__).parent / "act_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                # Skip _comment key
                self._act_mapping = {k: v for k, v in raw.items() if not k.startswith("_")}
            logger.info(f"KB: Loaded {len(self._act_mapping)} act name mappings")

        # Initialize ChromaDB (if available)
        chroma_dir = self.kb_dir / "chroma"
        if chroma_dir.exists():
            try:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.PersistentClient(
                    path=str(chroma_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                self._chroma_collection = client.get_collection("thai_law_sections")
                logger.info(f"KB: ChromaDB loaded with {self._chroma_collection.count()} vectors")
            except Exception as e:
                logger.warning(f"KB: ChromaDB not available: {e}")

        # Initialize Cohere client (for query embedding)
        cohere_key = os.environ.get("COHERE_API_KEY", "")
        if cohere_key:
            try:
                import cohere
                self._cohere_client = cohere.ClientV2(api_key=cohere_key)
                logger.info("KB: Cohere client initialized")
            except Exception as e:
                logger.warning(f"KB: Cohere not available: {e}")

        # KB is available if we have at least the section index
        self._available = bool(self._section_index)
        if self._available:
            logger.info("KB: Knowledge base is AVAILABLE")
        else:
            logger.warning("KB: Knowledge base is NOT available")

    def is_available(self) -> bool:
        """Check if the knowledge base is loaded and usable."""
        return self._available

    def has_semantic_search(self) -> bool:
        """Check if semantic search (ChromaDB + Cohere) is available."""
        return self._chroma_collection is not None and self._cohere_client is not None

    @property
    def act_count(self) -> int:
        return len(self._section_index)

    @property
    def section_count(self) -> int:
        return sum(len(sections) for sections in self._section_index.values())

    @property
    def build_date(self) -> str:
        return self._meta.get("build_date", "unknown")

    @property
    def source(self) -> str:
        return self._meta.get("source", "unknown")

    # ─── Stage 1: Exact Match ─────────────────────────────────────────────

    def _resolve_act_name(self, english_name: str) -> Optional[str]:
        """Resolve an English act name to its Thai equivalent."""
        if not english_name:
            return None

        # Normalize: lowercase, strip B.E. year suffix for lookup
        normalized = english_name.strip().lower()

        # Direct lookup
        if normalized in self._act_mapping:
            return self._act_mapping[normalized]

        # Try stripping " b.e. NNNN" suffix
        import re
        stripped = re.sub(r'\s*b\.?e\.?\s*\d{4}', '', normalized).strip()
        if stripped in self._act_mapping:
            return self._act_mapping[stripped]

        # Try partial match (contains)
        for eng, thai in self._act_mapping.items():
            if eng in normalized or normalized in eng:
                return thai

        return None

    def exact_lookup(self, act_name_en: str, section: str) -> Optional[dict]:
        """
        Stage 1: Look up an exact act name + section number.
        Returns dict with thai_text, act_name_thai, section if found.
        """
        if not self._available or not act_name_en:
            return None

        thai_name = self._resolve_act_name(act_name_en)
        if not thai_name:
            return None

        # Search through index keys for a match
        # (index keys are full Thai act titles which may include more detail)
        matched_act = None
        for index_key in self._section_index:
            if thai_name in index_key:
                matched_act = index_key
                break

        if not matched_act:
            return None

        act_sections = self._section_index[matched_act]

        # Normalize section number
        section_norm = str(section).strip()

        # Direct match
        if section_norm in act_sections:
            return {
                "thai_text": act_sections[section_norm],
                "act_name_thai": matched_act,
                "section": section_norm,
                "match_type": "exact",
            }

        # Try without spaces (e.g., "19 ทวิ" vs "19ทวิ")
        for sec_key, sec_text in act_sections.items():
            if sec_key.replace(" ", "") == section_norm.replace(" ", ""):
                return {
                    "thai_text": sec_text,
                    "act_name_thai": matched_act,
                    "section": sec_key,
                    "match_type": "exact",
                }

        return None

    # ─── Stage 2: Semantic Search ─────────────────────────────────────────

    async def semantic_search(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Stage 2: Semantic similarity search via ChromaDB + Cohere.
        Returns list of {text, act_title, section_number, distance}.
        """
        if not self.has_semantic_search() or not query_text:
            return []

        try:
            # Embed the query
            response = self._cohere_client.embed(
                texts=[query_text],
                model="embed-multilingual-v3.0",
                input_type="search_query",
                embedding_types=["float"],
            )
            query_embedding = response.embeddings.float_[0]

            # Query ChromaDB
            results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return []

            matches = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 1.0
                matches.append({
                    "text": doc,
                    "act_title": meta.get("act_title", ""),
                    "section_number": meta.get("section_number", ""),
                    "amendment_info": meta.get("amendment_info", ""),
                    "distance": dist,
                })

            return matches

        except Exception as e:
            logger.error(f"KB semantic search error: {e}")
            return []

    # ─── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return KB status for the /api/kb-status endpoint."""
        return {
            "available": self._available,
            "semantic_search": self.has_semantic_search(),
            "act_count": self.act_count,
            "section_count": self.section_count,
            "source": self.source,
            "build_date": self.build_date,
            "vector_count": self._chroma_collection.count() if self._chroma_collection else 0,
        }
