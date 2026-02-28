from dataclasses import dataclass

DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = (
    "mlx-community/gemma-3n-e4b-it-4bit",
    "mlx-community/gemma-3n-E4B-it-4bit",
    "mlx-community/gemma-3n-e4b-it",
)
FALLBACK_MODEL = "mlx-community/gemma-3-4b-it-4bit"


@dataclass(slots=True)
class AgentConfig:
    model: str | None = None
    camera_interval: float = 4.0
    max_tokens: int = 256
    temperature: float = 0.1
    ignore_people: bool = False
    default_model_candidates: tuple[str, ...] = DEFAULT_MODEL_CANDIDATES
    fallback_model: str = FALLBACK_MODEL
