import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


class BaseChunker:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def _chunk_text_by_pattern(
            self, text: str, pattern: str, file_path: str
        ) -> List[Dict[str, Any]]:
        chunks = []
        matches = list(re.finditer(pattern, text))

        start_idx = 0
        for i in range(len(matches) + 1):
            end_idx = matches[i].start() if i < len(matches) else len(text)
            raw_chunk = text[start_idx:end_idx]
            clean_chunk = raw_chunk.strip()

            if not clean_chunk:
                if i < len(matches):
                    start_idx = matches[i].end()
                continue

            offset = len(raw_chunk) - len(raw_chunk.lstrip())
            current_start = start_idx + offset

            while len(clean_chunk) > self.max_chunk_size:
                chunks.append({
                    "file_path": file_path,
                    "first_character_index": current_start,
                    "last_character_index": current_start + self.max_chunk_size
                })
                current_start += self.max_chunk_size
                clean_chunk = clean_chunk[self.max_chunk_size:]

            if clean_chunk:
                chunks.append({
                    "file_path": file_path,
                    "first_character_index": current_start,
                    "last_character_index": current_start + len(clean_chunk)
                })

            if i < len(matches):
                start_idx = matches[i].end()
        
        return chunks


class MarkdownChunker(BaseChunker):
    def chunk(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        pattern = r'\n\n+|\n(?=# )'
        return self._chunk_text_by_pattern(text, pattern, file_path)


class PythonChunker(BaseChunker):
    def chunk(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        pattern = r'\n(?=def | \s*class )'
        return self._chunk_text_by_pattern(text, pattern, file_path)


class Indexer:
    def __init__(
            self, raw_dir: str, processed_dir: str,
            max_chunk_size: int = 2000
        ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path("data/processed")
        self.max_chunk_size = max_chunk_size
        self.md_chunker = MarkdownChunker(max_chunk_size)
        self.py_chunker = PythonChunker(max_chunk_size)

    def build_index(self) -> None:
        all_chunks = []
        files_to_process = []

        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(('.py', '.md')):
                    files_to_process.append(Path(root) / file)

        for file_path in tqdm(files_to_process, desc="Chunking Files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                continue

            relative_path = str(file_path)
            if file_path.suffix == '.md':
                chunks = self.md_chunker.chunk(text, relative_path)
            elif file_path.suffix == '.py':
                chunks = self.py_chunker.chunk(text, relative_path)

            all_chunks.extend(chunks)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.processed_dir / "index.json"

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)

        print(
            f"\nIngestion complete! Indexed {len(all_chunks)} "
            f"chunks under {self.processed_dir}/"
        )
