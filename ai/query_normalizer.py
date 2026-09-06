"""Query normalizer using a local Llama 3.2 1B GGUF model.

Converts natural-language user queries into concise visual search
phrases suitable for an open-vocabulary object detector.
"""

import logging
import re
import time
from pathlib import Path

from llama_cpp import Llama

from ai.prompts import QUERY_NORMALIZER_SYSTEM_PROMPT
from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)

# Validation thresholds
_MAX_NORMALIZED_LENGTH: int = 500
_MAX_INPUT_LENGTH: int = 1000

# Phrases that indicate the SLM echoed instructions instead of extracting
_INVALID_PREFIXES: tuple[str, ...] = (
    "i ",
    "you ",
    "the user ",
    "sure",
    "here",
    "output:",
    "answer:",
)


class QueryNormalizer:
    """Normalizes natural-language queries into visual search phrases.

    Uses a local GGUF model for extraction.  The model is loaded once
    and remains in memory for the application lifetime.
    """

    def __init__(self) -> None:
        model_path: Path = (
            Path(__file__).parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {model_path}. "
                f"Place Llama-3.2-1B-Instruct-Q4_K_M.gguf in ai/models/"
            )

        logger.info("Loading query normalizer model from %s", model_path)
        start: float = time.perf_counter()

        self._model: Llama = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False,
        )

        duration: float = time.perf_counter() - start
        logger.info("Query normalizer model loaded in %.2fs (CPU mode)", duration)

    def normalize(self, user_query: str) -> Result:
        """Convert a natural-language query into a visual search phrase.

        Returns Result with the normalized query string on success,
        or Result with err if normalization fails.
        """
        assert isinstance(user_query, str), "user_query must be a string"

        stripped_query: str = user_query.strip()
        if not stripped_query:
            return Result(err="Query is empty")
        if len(stripped_query) > _MAX_INPUT_LENGTH:
            return Result(err="Query is too long")

        # A single noun is already a valid detector phrase. Skipping the SLM
        # prevents a small model from replacing the object with an invented
        # attribute, such as turning "bottle" into "white".
        if re.fullmatch(r"[\w-]+", stripped_query):
            return Result(result=stripped_query)

        logger.info("Normalizing query: '%s'", stripped_query)
        start: float = time.perf_counter()

        try:
            output = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": QUERY_NORMALIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": stripped_query},
                ],
                temperature=0.0,
                max_tokens=100,
            )
        except RuntimeError as e:
            logger.error("SLM generation failed: %s", e)
            return Result(err=f"Query processing failed: {e}")
        except ValueError as e:
            logger.error("SLM input error: %s", e)
            return Result(err=f"Invalid query input: {e}")

        duration: float = time.perf_counter() - start

        raw_output: str = output["choices"][0]["message"]["content"]
        normalized: str = _validate_output(raw_output)

        if not normalized:
            logger.warning(
                "SLM produced unusable output for '%s': '%s'",
                stripped_query,
                raw_output,
            )
            return Result(err="Query processing failed: model produced unusable output")

        logger.info(
            "Query normalized in %.2fs: '%s' → '%s'",
            duration,
            stripped_query,
            normalized,
        )
        return Result(result=normalized)


def _validate_output(raw_output: str) -> str:
    """Validate and clean SLM output.

    Returns the cleaned string, or empty string if invalid.
    """
    cleaned: str = raw_output.strip()

    # Remove accidental surrounding quotes
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    if not cleaned:
        return ""

    if len(cleaned) > _MAX_NORMALIZED_LENGTH:
        return ""

    # Reject outputs that look like instructions or conversational responses
    lower: str = cleaned.lower()
    for prefix in _INVALID_PREFIXES:
        if lower.startswith(prefix):
            return ""

    return cleaned
