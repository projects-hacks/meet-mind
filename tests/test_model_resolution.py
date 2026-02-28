import unittest
from unittest.mock import Mock

from backend.agents.roomscribe.agent import resolve_model_id
from backend.agents.roomscribe.config import DEFAULT_MODEL_CANDIDATES, FALLBACK_MODEL


class ResolveModelIdTests(unittest.TestCase):
    def test_default_first_candidate_succeeds(self) -> None:
        load_fn = Mock(return_value=("m", "p"))

        model_id, model, processor = resolve_model_id(
            user_model=None,
            default_model_candidates=DEFAULT_MODEL_CANDIDATES,
            fallback_model=FALLBACK_MODEL,
            load_fn=load_fn,
        )

        self.assertEqual(model_id, DEFAULT_MODEL_CANDIDATES[0])
        self.assertEqual(model, "m")
        self.assertEqual(processor, "p")
        load_fn.assert_called_once_with(DEFAULT_MODEL_CANDIDATES[0])

    def test_default_third_candidate_succeeds(self) -> None:
        calls = {"count": 0}

        def load_fn(model_id: str):
            calls["count"] += 1
            if model_id in DEFAULT_MODEL_CANDIDATES[:2]:
                raise RuntimeError("load failed")
            return "m", "p"

        model_id, _, _ = resolve_model_id(
            user_model=None,
            default_model_candidates=DEFAULT_MODEL_CANDIDATES,
            fallback_model=FALLBACK_MODEL,
            load_fn=load_fn,
        )

        self.assertEqual(model_id, DEFAULT_MODEL_CANDIDATES[2])
        self.assertEqual(calls["count"], 3)

    def test_default_fallback_succeeds(self) -> None:
        def load_fn(model_id: str):
            if model_id != FALLBACK_MODEL:
                raise RuntimeError("load failed")
            return "m", "p"

        model_id, _, _ = resolve_model_id(
            user_model=None,
            default_model_candidates=DEFAULT_MODEL_CANDIDATES,
            fallback_model=FALLBACK_MODEL,
            load_fn=load_fn,
        )

        self.assertEqual(model_id, FALLBACK_MODEL)

    def test_default_all_fail_raises_with_attempted_ids(self) -> None:
        def load_fn(_model_id: str):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            resolve_model_id(
                user_model=None,
                default_model_candidates=DEFAULT_MODEL_CANDIDATES,
                fallback_model=FALLBACK_MODEL,
                load_fn=load_fn,
            )

        msg = str(ctx.exception)
        for model_id in (*DEFAULT_MODEL_CANDIDATES, FALLBACK_MODEL):
            self.assertIn(model_id, msg)

    def test_explicit_model_is_strict(self) -> None:
        load_fn = Mock(return_value=("m", "p"))

        model_id, _, _ = resolve_model_id(
            user_model="custom/model",
            default_model_candidates=DEFAULT_MODEL_CANDIDATES,
            fallback_model=FALLBACK_MODEL,
            load_fn=load_fn,
        )

        self.assertEqual(model_id, "custom/model")
        load_fn.assert_called_once_with("custom/model")


if __name__ == "__main__":
    unittest.main()
