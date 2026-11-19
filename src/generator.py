import torch
from tqdm import tqdm
from typing import Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)


class QwenGenerator:
    """
    Connects to a local vLLM server running the Qwen model.
    Extracts text from raw files using indices, builds the prompt,
    and generates answers.
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-0.6B",
            max_new_tokens: int = 200,
    ) -> None:
        """Load tokenizer and model onto the available device."""

        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def _read_context(self, sources: List[MinimalSource]) -> str:
        """Extract and join text snippets from retrieved sources."""
        snippets: List[str] = []
        for idx, src in enumerate(sources, start=1):
            try:
                with open(src.file_path, "r", encoding="utf-8") as f:
                    text = f.read()[
                        src.first_character_index:src.last_character_index
                    ]
                if text:
                    snippets.append(f"[{idx}] {src.file_path}\n{text}")
            except OSError:
                continue

        return "\n\n".join(snippets)

    def _answer_one(
            self, question: str, sources: List[MinimalSource]
    ) -> str:
        """Generate a single grounded answer from retrieved context."""
        if not question.strip():
            return "Empty question."
        context = self._read_context(sources)
        if not context:
            return "Answer not found in the provided documents."

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
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        return str(
            self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        ).strip()

    def generate(
            self, search_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Generate answers for all retrieved search results."""
        answers: List[MinimalAnswer] = []
        for item in tqdm(search_results.search_results, desc="Generating"):
            try:
                text = self._answer_one(item.question, item.retrieved_sources)
            except Exception as exc:
                text = f"Error generating answer: {exc}"

            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=text,
                )
            )

        return StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k
        )
