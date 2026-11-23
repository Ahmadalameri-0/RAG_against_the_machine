import sys
from typing import List, Optional, Sequence, Tuple
from .models import MinimalSource

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RST = "\033[0m"

TQDM_FMT: Optional[str] = (
    "\033[1;36m{l_bar}\033[1;32m{bar}\033[0m{r_bar}"
)


def bar(ratio: float, width: int = 20) -> str:
    """Render a compact Unicode progress bar for a ratio in [0, 1]."""
    filled = int(round(max(0.0, min(1.0, ratio)) * width))
    return "█" * filled + "░" * (width - filled)


def header(title: str) -> None:
    """Print a Unicode box header."""
    line = "─" * 58
    print(f"{CYAN}┌{line}┐{RST}")
    print(f"{CYAN}│{RST} {BOLD}{title:<56}{RST} {CYAN}│{RST}")
    print(f"{CYAN}└{line}┘{RST}")


def ok(label: str, detail: str) -> None:
    """Print a green success line followed by a highlighted detail."""
    print(f"{GREEN}✔ {label}{RST} -> {CYAN}{detail}{RST}")


def warn(message: str) -> None:
    """Print a yellow warning on stderr."""
    print(f"{YELLOW}{message}{RST}", file=sys.stderr)


def error(message: str) -> None:
    """Print a red error on stderr."""
    print(f"{RED}Error:{RST} {message}", file=sys.stderr)


def _sources(items: Sequence[MinimalSource]) -> None:
    """Print ranked source locations."""
    for i, s in enumerate(items, start=1):
        span = f"[{s.first_character_index}:{s.last_character_index}]"
        print(f"  {CYAN}#{i:<2}{RST} {s.file_path} {DIM}{span}{RST}")


def search_result(query: str, items: Sequence[MinimalSource]) -> None:
    """Print the result of a single-query search."""
    print(f"{BOLD}Query:{RST} {query}")
    if not items:
        print(f"{YELLOW}No sources retrieved.{RST}")
        return
    _sources(items)


def answer_result(
        question: str, answer: str, items: Sequence[MinimalSource]
) -> None:
    """Print the result of a single-query answer."""
    print(f"{BOLD}Question:{RST} {question}")
    print(f"{GREEN}{BOLD}Answer  :{RST} {answer}")
    print(f"{BOLD}Sources :{RST}")
    _sources(items)


def recall_report(
        name: str,
        total: int,
        with_ret: int,
        k: int,
        results: List[Tuple[int, float]],
        target: float,
        max_span: int,
        iou: float,
) -> None:
    """Print the Recall@k table; Recall@5 is checked against target."""
    header(f"EVALUATION REPORT — {name}")
    print(
        f"  Questions : {total} evaluated "
        f"| {with_ret} with retrieved sources | k={k}"
    )
    print(
        f"  Validity  : {GREEN}OK{RST} "
        f"(all spans <= {max_span} chars, IoU > {iou})\n"
    )
    print(f"  {'Metric':<10} │ {'Recall %':>8} │ {'Progress':<20} │ Score")
    print(f"  {'─' * 10}─┼─{'─' * 8}─┼─{'─' * 20}─┼─{'─' * 6}")
    for n, avg in results:
        color = GREEN if avg >= 0.8 else (YELLOW if avg >= 0.5 else RED)
        status = ""
        if n == 5:
            tag = f"{GREEN}PASS{RST}" if avg >= target else f"{RED}FAIL{RST}"
            status = f"  [{tag} >= {target * 100:.0f}%]"
        label = f"Recall@{n}"
        print(
            f"  {BOLD}{label:<10}{RST} │ "
            f"{color}{avg * 100:>7.1f}%{RST} │ "
            f"{color}{bar(avg)}{RST} │ {avg:.3f}{status}"
        )
