"""MLX-based Gemma model provider — handles all LLM inference on Apple Silicon."""

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from mlx_lm import load, generate  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 0.5


def _is_local_model_reference(model_id: str) -> bool:
    """True when model_id points to a local path or local file reference."""
    if not model_id or not model_id.strip():
        return False
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return True
    # Explicit local-file convention
    return model_id.startswith("file://")


def _extract_json(raw: str) -> dict[str, Any]:
    """Robustly extract JSON from model output.

    Handles: markdown fences, trailing text, nested braces, truncated output,
    trailing commas, missing closing braces.
    """
    if not raw or not raw.strip():
        return {}

    text = raw.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]

    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-matching: find the outermost complete JSON object
    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Truncation recovery: fix trailing comma + close missing braces
    candidate = text[start:]
    # Remove trailing incomplete values
    candidate = candidate.rstrip()
    for _ in range(5):  # Try progressively closing braces
        candidate = candidate.rstrip(",").rstrip()
        if not candidate.endswith("}"):
            candidate += "}"
        # Close arrays too
        open_brackets = candidate.count("[") - candidate.count("]")
        candidate += "]" * max(0, open_brackets)
        open_braces = candidate.count("{") - candidate.count("}")
        candidate += "}" * max(0, open_braces)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Strip last key-value pair and try again
            last_comma = candidate.rfind(",")
            if last_comma > 0:
                candidate = candidate[:last_comma]
            else:
                break

    logger.warning(f"JSON extraction failed: {raw[:300]}")
    return {}


class GemmaMLX:
    """Wraps mlx-lm for Gemma inference. Implements the LLMProvider protocol.

    Features:
    - Lazy model loading with error handling
    - Retry logic for transient failures
    - Robust JSON extraction and truncation recovery
    """

    def __init__(
        self,
        model_id: str = "mlx-community/gemma-3-4b-it-4bit",
        allow_remote_models: bool = True,
    ):
        self._model_id = model_id
        self._allow_remote_models = allow_remote_models
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._load_error:
            raise RuntimeError(f"Model previously failed to load: {self._load_error}")
        if not self._allow_remote_models and not _is_local_model_reference(self._model_id):
            raise RuntimeError(
                f"Remote model reference not allowed in current policy: {self._model_id}. "
                "Use a local model path."
            )
        try:
            logger.info(f"Loading model: {self._model_id}")
            self._model, self._tokenizer = load(self._model_id)
            logger.info(f"Model loaded: {self._model_id}")
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Failed to load model {self._model_id}: {e}")
            raise

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        """Generate text with retry logic."""
        self._ensure_loaded()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = generate(
                    self._model, self._tokenizer,
                    prompt=formatted,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                return response.strip()
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Generation attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Generation failed after {MAX_RETRIES+1} attempts: {e}")
                    return ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Generate and parse JSON output with retry on parse failure."""
        for attempt in range(MAX_RETRIES + 1):
            raw = self.generate(prompt, system_prompt, max_tokens)
            if not raw:
                continue
            result = _extract_json(raw)
            if result:
                return result
            if attempt < MAX_RETRIES:
                logger.info(f"JSON parse retry {attempt+1}: re-generating...")
                time.sleep(RETRY_DELAY)
        return {}

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


@lru_cache(maxsize=8)
def get_model(model_id: str, allow_remote_models: bool = True) -> GemmaMLX:
    """Factory — caches model instances so we don't reload."""
    return GemmaMLX(model_id, allow_remote_models=allow_remote_models)
