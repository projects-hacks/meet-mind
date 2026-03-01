"""Gemma model provider — supports LM Studio API and MLX backends.

Backends:
  - LM Studio: Calls localhost API (http://localhost:1234/v1/chat/completions)
    Best for fine-tuned models loaded in LM Studio GUI.
  - MLX: Direct Apple Silicon inference via mlx-lm library.
    Best for quick prototyping with HuggingFace models.

Memory management:
  - Singleton cache: same model_id → same instance (avoids duplicate loads)
  - unload() support: explicitly free model weights + clear Metal cache
  - ModelPool: coordinates VLM ↔ text model lifecycle for memory-constrained devices
"""

import gc
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 0.5

# ── Global GPU inference lock ──────────────────────────────────────────
# Apple Silicon has ONE unified GPU. Concurrent MLX inference (text + VLM)
# causes [METAL] Command buffer execution failed: Caused GPU Timeout Error.
# ALL mlx-lm and mlx-vlm generate() calls MUST acquire this lock first.
mlx_inference_lock = threading.Lock()

# LM Studio default endpoint
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")


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
    candidate = candidate.rstrip()
    for _ in range(5):
        candidate = candidate.rstrip(",").rstrip()
        if not candidate.endswith("}"):
            candidate += "}"
        open_brackets = candidate.count("[") - candidate.count("]")
        candidate += "]" * max(0, open_brackets)
        open_braces = candidate.count("{") - candidate.count("}")
        candidate += "}" * max(0, open_braces)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            last_comma = candidate.rfind(",")
            if last_comma > 0:
                candidate = candidate[:last_comma]
            else:
                break

    logger.warning(f"JSON extraction failed: {raw[:300]}")
    return {}


# ══════════════════════════════════════════════════════════════════════
# LM Studio API Provider (Primary — for fine-tuned Gemma 3 4B)
# ══════════════════════════════════════════════════════════════════════

class GemmaLMStudio:
    """Calls LM Studio's OpenAI-compatible API for inference.

    LM Studio runs locally and serves the fine-tuned Gemma 3 4B model.
    API: http://localhost:1234/v1/chat/completions
    """

    def __init__(self, api_url: str = LM_STUDIO_URL):
        self._api_url = api_url
        self._model_name = None  # Auto-detected from LM Studio

    def _get_model_name(self) -> str:
        """Auto-detect the loaded model name from LM Studio."""
        if self._model_name:
            return self._model_name
        try:
            models_url = self._api_url.replace("/chat/completions", "/models")
            resp = requests.get(models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    self._model_name = data["data"][0]["id"]
                    return self._model_name
        except Exception:
            pass
        return "local-model"

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        """Generate text via LM Studio API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._api_url,
                    json={
                        "model": self._get_model_name(),
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"LM Studio attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"LM Studio failed after {MAX_RETRIES+1} attempts: {e}")
                    return ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Generate and parse JSON output with retry."""
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
        return f"lmstudio:{self._get_model_name()}"

    @property
    def is_loaded(self) -> bool:
        try:
            models_url = self._api_url.replace("/chat/completions", "/models")
            resp = requests.get(models_url, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════
# MLX Provider (Fallback — direct Apple Silicon inference)
# ══════════════════════════════════════════════════════════════════════

class GemmaMLX:
    """Wraps mlx-lm for Gemma inference on Apple Silicon.

    Features:
    - Lazy model loading with error handling
    - Retry logic for transient failures
    - Robust JSON extraction and truncation recovery
    - Explicit unload() for memory management
    """

    def __init__(
        self,
        model_id: str = "mlx-community/gemma-3-4b-it-4bit",
        allow_remote_models: bool = True,
    ):
        self._model_id = model_id
        self._allow_remote_models = allow_remote_models
        self._model = None
        self._tokenizer = None
        self._load_error = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self._load_error:
                # Allow retry after explicit unload()
                pass
            if not self._allow_remote_models and not Path(self._model_id).expanduser().exists():
                raise RuntimeError(f"Remote model not allowed: {self._model_id}")
            try:
                from mlx_lm import load
                logger.info(f"Loading MLX model: {self._model_id}")
                self._model, self._tokenizer = load(self._model_id)
                self._load_error = None
                logger.info(f"MLX model loaded: {self._model_id}")
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"Failed to load MLX model {self._model_id}: {e}")
                raise

    def unload(self):
        """Explicitly free model weights and clear Metal cache.

        After unload(), the next generate() call will lazy-reload.
        """
        with self._lock:
            if self._model is None:
                return
            model_id = self._model_id
            self._model = None
            self._tokenizer = None
            self._load_error = None
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass
            logger.info(f"MLX model unloaded: {model_id} (memory freed)")

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        """Generate text with retry logic. Acquires GPU lock to prevent Metal timeout."""
        self._ensure_loaded()
        from mlx_lm import generate as mlx_generate

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                with mlx_inference_lock:
                    response = mlx_generate(
                        self._model, self._tokenizer,
                        prompt=formatted,
                        max_tokens=max_tokens,
                        verbose=False,
                    )
                return response.strip()
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"MLX attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"MLX failed after {MAX_RETRIES+1} attempts: {e}")
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


# ══════════════════════════════════════════════════════════════════════
# Factory — auto-selects best backend (with singleton cache)
# ══════════════════════════════════════════════════════════════════════

_model_cache: dict[str, GemmaLMStudio | GemmaMLX] = {}
_cache_lock = threading.Lock()

# Backend choice is locked at first detection to prevent mid-session switching.
# e.g. if LM Studio is running on first call but stopped later, reset() won't
# accidentally switch agents to MLX while others remain on LM Studio.
_backend_choice: str | None = None  # "lmstudio" | "mlx" | None (not yet decided)


def _detect_backend(forced: str | None) -> str:
    """Detect which backend to use. Result is locked for the session."""
    global _backend_choice

    if forced in ("lmstudio", "mlx"):
        return forced

    if _backend_choice is not None:
        return _backend_choice

    if _lmstudio_available():
        _backend_choice = "lmstudio"
        logger.info("Backend detected: LM Studio (locked for session)")
    else:
        _backend_choice = "mlx"
        logger.info("Backend detected: MLX (locked for session)")

    return _backend_choice


def get_model(
    model_id: str = None,
    backend: str = None,
    allow_remote_models: bool = True,
) -> GemmaLMStudio | GemmaMLX:
    """Factory — returns the best available model provider.

    IMPORTANT: Returns a cached singleton for the same model_id.
    This prevents loading the same ~3.5 GB model 3 times for Scribe/Analyst/Architect.

    Backend choice is detected once and locked for the entire session to prevent
    inconsistent behavior if LM Studio starts/stops mid-meeting.

    Backend priority:
      1. 'lmstudio' — if LM Studio is running, use it (fine-tuned model)
      2. 'mlx' — fallback to direct MLX inference (singleton per model_id)

    Args:
        model_id: Model identifier (HuggingFace ID or local path)
        backend: Force 'lmstudio' or 'mlx'. None = auto-detect.
        allow_remote_models: Whether MLX can download remote models.
    """
    model_id = model_id or "mlx-community/gemma-3-4b-it-4bit"

    # Fast path: return cached instance immediately (no network I/O)
    with _cache_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]

    # Slow path: detect backend (only hits network on the very first call)
    chosen = _detect_backend(backend)

    with _cache_lock:
        # Double-check after re-acquiring lock
        if model_id in _model_cache:
            return _model_cache[model_id]

        if chosen == "lmstudio":
            instance: GemmaLMStudio | GemmaMLX = GemmaLMStudio()
            logger.info(f"Created LM Studio instance (cached as '{model_id}')")
        else:
            instance = GemmaMLX(model_id, allow_remote_models=allow_remote_models)
            logger.info(f"Created MLX model instance (cached): {model_id}")

        _model_cache[model_id] = instance
        return instance


def unload_all_models():
    """Unload all cached models to free memory."""
    with _cache_lock:
        for mid, model in _model_cache.items():
            if isinstance(model, GemmaMLX):
                model.unload()
        logger.info(f"Unloaded {len(_model_cache)} cached model(s)")


def unload_model(model_id: str):
    """Unload a specific cached model by ID."""
    with _cache_lock:
        model = _model_cache.get(model_id)
        if model is not None and isinstance(model, GemmaMLX):
            model.unload()
            logger.info(f"Unloaded model: {model_id}")


def _lmstudio_available() -> bool:
    """Check if LM Studio is running and has a model loaded."""
    try:
        url = LM_STUDIO_URL.replace("/chat/completions", "/models")
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False
