import json
import requests
from typing import List
from .models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    MinimalSource
)


class QwenGenerator:
    """
    Connects to a local vLLM server running the Qwen model.
    Extracts text from raw files using indices, builds the prompt,
    and generates answers.
    """