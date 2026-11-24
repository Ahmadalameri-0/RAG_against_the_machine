import json
import os
import re
from typing import Any, Dict, List
from tqdm import tqdm
from .chunker import Chunker
from collections import Counter
from .CLI import TQDM_FMT


class Indexer:
    """Ingest raw files, chunk them, and build a BM25 inverted index."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Initialize indexer with configurable input/output directories."""
        self.raw_dir: str = raw_dir
        self.processed_dir: str = processed_dir
        self.chunker: Chunker = Chunker()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text while preserving code identifiers and sub-words."""
        tokens: List[str] = []
        for raw in re.findall(r"[A-Za-z0-9]+", text):
            lower = raw.lower()
            tokens.append(lower)
            parts = re.sub(r"([a-z])([A-Z])", r"\1_\2", raw).split("_")
            if len(parts) > 1:
                for part in parts:
                    p_low = part.lower()
                    if len(p_low) > 1 and p_low != lower:
                        tokens.append(p_low)
        return tokens

    def _collect_files(self) -> List[str]:
        """Collect supported source and documentation files"""
        exts = (".py", ".md", ".txt")
        skip_dirs = {".git", "__pycache__", ".venv", "node_modules"}
        files: List[str] = []
        for root, dirs, filenames in os.walk(self.raw_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in filenames:
                if name.endswith(exts):
                    full_path = os.path.join(root, name).replace("\\", "/")
                    files.append(full_path)
        files.sort()
        return files

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk corpus and persist compact inverted index to disk"""
        files = self._collect_files()
        chunks: List[Dict[str, Any]] = []
        inverted_index: Dict[str, List[List[int]]] = {}
        total_len: int = 0

        for file_path in tqdm(files, desc="Indexing", bar_format=TQDM_FMT):
            try:
                with open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = f.read()
            except OSError:
                continue

            if file_path.endswith(".py"):
                spans = self.chunker.chunk_python(content, max_chunk_size)
            else:
                spans = self.chunker.chunk_markdown(content, max_chunk_size)

            path_tokens = self.tokenize(file_path)
            for start, end in spans:
                doc_tokens = path_tokens + self.tokenize(content[start:end])
                if not doc_tokens:
                    continue
                doc_id = len(chunks)
                doc_len = len(doc_tokens)
                total_len += doc_len
                chunks.append(
                    {
                        "file_path": file_path,
                        "first_character_index": start,
                        "last_character_index": end,
                        "length": doc_len
                    }
                )
                for term, tf in Counter(doc_tokens).items():
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append([doc_id, tf])

        os.makedirs(self.processed_dir, exist_ok=True)
        index_path = os.path.join(self.processed_dir, "index.json")
        avg_len = (total_len / len(chunks)) if chunks else 0.0
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            f.write(f'  "num_docs": {len(chunks)},\n')
            f.write(f'  "avg_doc_len": {avg_len},\n')
            f.write('  "chunks": [\n')
            for i, c in enumerate(chunks):
                comma = "," if i < len(chunks) - 1 else ""
                line = json.dumps(c, separators=(",", ":"))
                f.write(f"    {line}{comma}\n")
            f.write("  ],\n")
            f.write('  "inverted_index": {\n')
            items = list(inverted_index.items())
            for i, (term, postings) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                k = json.dumps(term)
                v = json.dumps(postings, separators=(",", ":"))
                f.write(f"    {k}: {v}{comma}\n")
            f.write("  }\n}\n")
