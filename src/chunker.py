import re
from typing import List, Tuple


class Chunker:
    """Split Python and Markdown/text content into character spans."""

    @staticmethod
    def _cut(content: str, pos: int, limit: int) -> int:
        """Return a cut <= pos + limit, preferring a newline, then a space"""
        for sep in ("\n", " "):
            cut = content.rfind(sep, pos + limit // 2, pos + limit)
            if cut != -1:
                return cut + 1
        return pos + limit

    def _fit_spans(
            self,
            content: str,
            raw_spans: List[Tuple[int, int]],
            max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Pack or split character spans so every chunk <= max_chunk_size"""
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
                    if end - pos <= max_chunk_size:
                        nxt = end
                    else:
                        nxt = self._cut(content, pos, max_chunk_size)
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

    def _chunk_by_patter(
            self, content: str, pattern: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Split content by regex boundaries keeping exact char offsets."""
        starts = [0] + [m.start() + 1 for m in re.finditer(pattern, content)]
        spans = [
            (starts[i], starts[i + 1]) for i in range(len(starts) - 1)
        ] + [(starts[-1], len(content))]
        return self._fit_spans(content, spans, max_chunk_size)

    def chunk_python(
            self, content: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Chunk Python files along class, def, and async def boundaries"""
        pattern = r"\n(?=\s*(?:async\s+def |def |class ))"
        return self._chunk_by_patter(content, pattern, max_chunk_size)

    def chunk_markdown(
            self, content: str, max_chunk_size: int
    ) -> List[Tuple[int, int]]:
        """Chunk Markdown/text files along headings and blank lines"""
        pattern = r"\n(?=#{1,6}\s)|\n\n"
        return self._chunk_by_patter(content, pattern, max_chunk_size)
