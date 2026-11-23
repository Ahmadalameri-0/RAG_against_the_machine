import json
import os
import re
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
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

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text while preserving code identifiers and sub-words."""
        tokens: List[str] = []
        for raw in re.findall(r"[A-Za-z0-9_]+", text):
            lower = raw.lower()
            tokens.append(lower)
            parts = re.sub(r"([a-z])([A-Z])", r"\1_\2", raw).split("_")
            if len(parts) > 1:
                for part in parts:
                    p_low = part.lower()
                    if len(p_low) > 1 and p_low != lower:
                        tokens.append(p_low)
        return tokens

    def _fit_spans(
        self,
        content: str,
        raw_spans: List[Tuple[int, int]],
        max_chunk_size: int,
    ) -> List[Tuple[int, int]]:
        """Pack or split character spans so every chunk <= max_chunk_size."""
        fitted: List[Tuple[int, int]] = []
        cur_start: int = -1
        cur_end: int = -1

        for start, end in raw_spans:
            if not content[start:end].strip():
                continue
            if end - start > max_chunk_size:
                if cur_start != -1:
                    fitted.append((cur_start, cur_end))
                    cur_start, cur_end = -1, -1
                pos = start
                while pos < end:
                    nxt = min(pos + max_chunk_size, end)
                    if content[pos:nxt].strip():
                        fitted.append((pos, nxt))
                    pos = nxt
            elif cur_start == -1:
                cur_start, cur_end = start, end
            elif end - cur_start <= max_chunk_size:
                cur_end = end
            else:
                fitted.append((cur_start, cur_end))
                cur_start, cur_end = start, end

        if cur_start != -1:
            fitted.append((cur_start, cur_end))
        return fitted

    def _chunk_by_pattern(
        self, content: str, pattern: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Split content by regex boundaries keeping exact char offsets."""
        starts = [0] + [m.start() + 1 for m in re.finditer(pattern, content)]
        spans = [
            (starts[i], starts[i + 1]) for i in range(len(starts) - 1)
        ] + [(starts[-1], len(content))]
        return self._fit_spans(content, spans, max_chunk_size)

    def _chunk_python(
        self, content: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Chunk Python files along class, def, and async def boundaries."""
        pattern = r"\n(?=\s*(?:async\s+def |def |class ))"
        return self._chunk_by_pattern(content, pattern, max_chunk_size)

    def _chunk_markdown(
        self, content: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Chunk Markdown/text files along headings and blank lines."""
        pattern = r"\n(?=#{1,6}\s)|\n\n"
        return self._chunk_by_pattern(content, pattern, max_chunk_size)

    def _collect_files(self) -> List[str]:
        """Collect supported source and documentation files."""
        exts = (".py", ".md", ".txt", ".rst")
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
        """Chunk corpus and persist compact inverted index to disk."""
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
                spans = self._chunk_python(content, max_chunk_size)
            else:
                spans = self._chunk_markdown(content, max_chunk_size)

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
                        "length": doc_len,
                    }
                )
                for term, tf in Counter(doc_tokens).items():
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append([doc_id, tf])

        os.makedirs(self.processed_dir, exist_ok=True)
        index_path = os.path.join(self.processed_dir, "index.json")
        avg_len = (total_len / len(chunks)) if chunks else 0.0
        payload: Dict[str, Any] = {
            "num_docs": len(chunks),
            "avg_doc_len": avg_len,
            "chunks": chunks,
            "inverted_index": inverted_index,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("{\n")
            for pos, (key, value) in enumerate(payload.items()):
                dumped = json.dumps(value, separators=(",", ":"))
                comma = "," if pos < len(payload) - 1 else ""
                f.write(f'  "{key}": {dumped}{comma}\n')
            f.write("}\n")
