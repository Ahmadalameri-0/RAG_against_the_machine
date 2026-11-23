import torch
from tqdm import tqdm
from typing import Any, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .CLI import TQDM_FMT
from .models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class QwenGenerator:
    """
    Local Qwen model wrapper (Hugging Face transformers).
    Extracts text from raw files using indices, builds prompts within a
    token budget, and generates answers in batches.
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-0.6B",
            max_new_tokens: int = 80,
            max_context_tokens: int = 800,
            batch_size: int = 2,
    ) -> None:
        """Load tokenizer and model onto the available device."""

        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.batch_size = max(1, batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        )
        self.model: Any = model.to(self.device)

    def _read_context(self, sources: List[MinimalSource]) -> str:
        """Extract snippets from sources, within the context token budget."""
        snippets: List[str] = []
        budget = self.max_context_tokens
        for idx, src in enumerate(sources, start=1):
            try:
                with open(
                    src.file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    text = f.read()[
                        src.first_character_index:src.last_character_index
                    ]
            except OSError:
                continue
            if not text:
                continue
            snippet = f"[{idx}] {src.file_path}\n{text}"
            ids = self.tokenizer.encode(snippet)
            if len(ids) > budget:
                snippet = self.tokenizer.decode(ids[:budget])
            snippets.append(snippet)
            budget -= len(ids)
            if budget <= 0:
                break
        return "\n\n".join(snippets)

    def _prepare(
            self, question: str, sources: List[MinimalSource]
    ) -> Tuple[Optional[str], str]:
        """Return (prompt, fallback); prompt is None if no model call."""
        if not question.strip():
            return None, "Empty question."
        context = self._read_context(sources)
        if not context:
            return None, "Answer not found in the provided documents."

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer concisely using ONLY the provided context. "
                    "If missing, say 'Answer not found in the provided "
                    "documents.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return str(prompt), ""

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate one grounded answer per prompt in a single batch."""
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        width = inputs.input_ids.shape[1]
        return [
            str(
                self.tokenizer.decode(seq[width:], skip_special_tokens=True)
            ).strip()
            for seq in out
        ]

    def generate(
            self, search_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Generate answers for all retrieved search results."""
        items = search_results.search_results
        texts: List[str] = [""] * len(items)
        pending: List[Tuple[int, str]] = []

        for i, item in enumerate(items):
            try:
                prompt, fallback = self._prepare(
                    item.question, item.retrieved_sources
                )
            except Exception as exc:
                prompt, fallback = None, f"Error generating answer: {exc}"
            if prompt is None:
                texts[i] = fallback
            else:
                pending.append((i, prompt))

        pending.sort(key=lambda pair: len(pair[1]))
        with tqdm(
            total=len(pending), desc="Generating", bar_format=TQDM_FMT
        ) as bar:
            for start in range(0, len(pending), self.batch_size):
                chunk = pending[start:start + self.batch_size]
                try:
                    outs = self._generate_batch([p for _, p in chunk])
                except Exception as exc:
                    outs = [f"Error generating answer: {exc}"] * len(chunk)
                for (i, _), text in zip(chunk, outs):
                    texts[i] = text
                bar.update(len(chunk))

        answers: List[MinimalAnswer] = [
            MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=texts[i],
            )
            for i, item in enumerate(items)
        ]
        return StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k
        )
