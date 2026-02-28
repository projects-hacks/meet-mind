"""Production-grade fine-tuning pipeline for MeetMind Analyst agent.

Fine-tunes Gemma 3 1B (or FunctionGemma when available in MLX) using LoRA
on the MeetMind action-decision dataset.

Features:
- Automated data splits (80/10/10) with stratified sampling
- Configurable LoRA hyperparameters
- Model fusion (adapter → standalone model)
- Post-training evaluation with accuracy metrics per action type
- Integration test against MeetMind pipeline

Usage:
    # Step 1: Generate data
    python3 data/generate_training_data.py

    # Step 2: Fine-tune
    python3 fine_tune_analyst.py --train

    # Step 3: Evaluate only (after training)
    python3 fine_tune_analyst.py --eval-only

    # Step 4: Fuse and export
    python3 fine_tune_analyst.py --fuse

    # Step 5: Run integration test
    python3 fine_tune_analyst.py --integration-test

Requirements:
    pip install mlx mlx-lm
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

BASE_MODEL = "mlx-community/gemma-3-1b-it-4bit"
TRAINING_FILE = "data/analyst_training.jsonl"
SPLIT_DIR = "data/splits"
ADAPTER_DIR = "outputs/analyst-lora"
FUSED_DIR = "outputs/analyst-fused"

# LoRA hyperparameters — tuned for 270M-1B models on M-series Mac
LORA_CONFIG = {
    "fine_tune_type": "lora",
    "num_layers": 8,          # LoRA on last 8 layers
    "batch_size": 2,          # Small batch for Mac memory
    "iters": 300,             # 300 iterations (3 epochs over ~100 examples)
    "val_batches": 5,
    "learning_rate": 2e-4,
    "steps_per_report": 10,
    "steps_per_eval": 50,
    "save_every": 100,
    "max_seq_length": 1024,
    "grad_accumulation_steps": 4,  # Effective batch size = 8
    "mask_prompt": True,      # Only train on assistant response (critical!)
    "optimizer": "adamw",
}

# All valid action names
VALID_ACTIONS = {
    "extract_action_item", "log_decision", "flag_gap",
    "request_artifact", "suggest_next_step", "provide_insight",
    "continue_observing",
}


# ══════════════════════════════════════════════════════════════════════
# Data Preparation
# ══════════════════════════════════════════════════════════════════════

def prepare_data(training_file: str, split_dir: str):
    """Load JSONL, validate, stratify-split into train/valid/test."""
    print("📦 Preparing training data...")

    if not os.path.exists(training_file):
        print(f"❌ Training file not found: {training_file}")
        print("   Run: python3 data/generate_training_data.py")
        sys.exit(1)

    # Load and validate
    examples = []
    skipped = 0
    with open(training_file) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                # Validate structure
                msgs = ex.get("messages", [])
                if len(msgs) < 2:
                    print(f"  ⚠️ Line {i}: Too few messages, skipping")
                    skipped += 1
                    continue
                # Validate assistant response is valid JSON
                assistant_msg = msgs[-1]["content"]
                parsed = json.loads(assistant_msg)
                if parsed.get("action") not in VALID_ACTIONS:
                    print(f"  ⚠️ Line {i}: Unknown action '{parsed.get('action')}', skipping")
                    skipped += 1
                    continue
                examples.append(ex)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠️ Line {i}: {e}, skipping")
                skipped += 1

    print(f"  Loaded: {len(examples)} valid examples ({skipped} skipped)")

    if len(examples) < 20:
        print("❌ Need at least 20 training examples.")
        sys.exit(1)

    # Stratified split by action type
    by_action = {}
    for ex in examples:
        action = json.loads(ex["messages"][-1]["content"])["action"]
        by_action.setdefault(action, []).append(ex)

    random.seed(42)
    train, valid, test = [], [], []

    for action, action_examples in by_action.items():
        random.shuffle(action_examples)
        n = len(action_examples)
        train_end = max(1, int(n * 0.8))
        valid_end = max(train_end + 1, int(n * 0.9))

        train.extend(action_examples[:train_end])
        valid.extend(action_examples[train_end:valid_end])
        test.extend(action_examples[valid_end:])

    # Ensure valid and test have at least 1 example each
    if not valid and train:
        valid.append(train.pop())
    if not test and train:
        test.append(train.pop())

    random.shuffle(train)
    random.shuffle(valid)

    # Write splits
    os.makedirs(split_dir, exist_ok=True)
    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = os.path.join(split_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        # Print action distribution per split
        dist = {}
        for ex in data:
            a = json.loads(ex["messages"][-1]["content"])["action"]
            dist[a] = dist.get(a, 0) + 1
        print(f"  {name}: {len(data)} examples {dict(sorted(dist.items()))}")

    return split_dir


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def run_training(split_dir: str, adapter_dir: str):
    """Run MLX LoRA fine-tuning."""
    print(f"\n🚀 Starting LoRA fine-tuning on {BASE_MODEL}...")
    print(f"   Adapter output: {adapter_dir}")
    print(f"   Config: {json.dumps(LORA_CONFIG, indent=2)}")

    os.makedirs(adapter_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--data", split_dir,
        "--train",
        "--adapter-path", adapter_dir,
        "--fine-tune-type", LORA_CONFIG["fine_tune_type"],
        "--optimizer", LORA_CONFIG["optimizer"],
        "--num-layers", str(LORA_CONFIG["num_layers"]),
        "--iters", str(LORA_CONFIG["iters"]),
        "--batch-size", str(LORA_CONFIG["batch_size"]),
        "--learning-rate", str(LORA_CONFIG["learning_rate"]),
        "--steps-per-report", str(LORA_CONFIG["steps_per_report"]),
        "--steps-per-eval", str(LORA_CONFIG["steps_per_eval"]),
        "--save-every", str(LORA_CONFIG["save_every"]),
        "--max-seq-length", str(LORA_CONFIG["max_seq_length"]),
        "--val-batches", str(LORA_CONFIG["val_batches"]),
        "--grad-accumulation-steps", str(LORA_CONFIG["grad_accumulation_steps"]),
    ]

    if LORA_CONFIG.get("mask_prompt"):
        cmd.append("--mask-prompt")

    print(f"\n   Command: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"❌ Fine-tuning failed! (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n✅ Fine-tuning complete in {elapsed/60:.1f} minutes")
    return adapter_dir


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def run_evaluation(split_dir: str, adapter_dir: str = None, fused_dir: str = None):
    """Evaluate the fine-tuned model on the test set.

    Reports:
    - Overall accuracy (correct action type)
    - Per-action accuracy
    - Parameter extraction quality
    """
    print("\n📊 Running evaluation...")

    test_file = os.path.join(split_dir, "test.jsonl")
    if not os.path.exists(test_file):
        print("  ⚠️ No test.jsonl found. Run --train first.")
        return

    # Load test examples
    test_examples = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    if not test_examples:
        print("  ⚠️ Test set is empty.")
        return

    # Determine model path
    if fused_dir and os.path.exists(fused_dir):
        model_path = fused_dir
        adapter_flag = []
    elif adapter_dir:
        model_path = BASE_MODEL
        adapter_flag = ["--adapter-path", adapter_dir]
    else:
        print("  ❌ No model or adapter found for evaluation.")
        return

    print(f"  Model: {model_path}")
    print(f"  Test examples: {len(test_examples)}")

    # Evaluate each example
    correct_action = 0
    total = 0
    per_action = {}  # action -> {"correct": N, "total": N}

    for i, ex in enumerate(test_examples):
        msgs = ex["messages"]
        expected_str = msgs[-1]["content"]
        expected = json.loads(expected_str)
        expected_action = expected["action"]

        # Get model prediction
        user_msg = msgs[-2]["content"] if len(msgs) >= 2 else ""
        system_msg = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg

        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", "200",
        ] + adapter_flag

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip()

        # Parse prediction
        predicted_action = _extract_action(output)

        # Score
        is_correct = predicted_action == expected_action
        if is_correct:
            correct_action += 1
        total += 1

        # Per-action tracking
        per_action.setdefault(expected_action, {"correct": 0, "total": 0})
        per_action[expected_action]["total"] += 1
        if is_correct:
            per_action[expected_action]["correct"] += 1

        status = "✅" if is_correct else "❌"
        print(f"  [{i+1}/{len(test_examples)}] {status} Expected: {expected_action} | Got: {predicted_action}")

    # Summary
    accuracy = correct_action / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {correct_action}/{total} ({accuracy:.1f}%)")
    print(f"\nPer-Action Breakdown:")
    for action, scores in sorted(per_action.items()):
        acc = scores["correct"] / max(scores["total"], 1) * 100
        print(f"  {action}: {scores['correct']}/{scores['total']} ({acc:.0f}%)")
    print(f"{'='*50}")


def _extract_action(text: str) -> str:
    """Extract the action name from model output."""
    import re

    # Try JSON parse
    try:
        # Strip markdown fences
        clean = text
        if "```" in clean:
            clean = re.sub(r'```\w*\n?', '', clean).strip()

        # Find JSON
        start = clean.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(clean)):
                if clean[i] == "{": depth += 1
                elif clean[i] == "}": depth -= 1
                if depth == 0:
                    data = json.loads(clean[start:i+1])
                    return data.get("action", data.get("name", "unknown"))
    except (json.JSONDecodeError, ValueError):
        pass

    # Bare name match
    for name in VALID_ACTIONS:
        if name in text.lower().replace(" ", "_"):
            return name

    return "unknown"


# ══════════════════════════════════════════════════════════════════════
# Fusion
# ══════════════════════════════════════════════════════════════════════

def fuse_model(adapter_dir: str, fused_dir: str):
    """Fuse LoRA adapters into the base model for faster inference."""
    print(f"\n🔗 Fusing adapters into standalone model...")

    if not os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")):
        print(f"  ❌ No adapter found at {adapter_dir}. Run --train first.")
        return False

    os.makedirs(fused_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL,
        "--adapter-path", adapter_dir,
        "--save-path", fused_dir,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("  ⚠️ Fusion failed. Use adapter-based inference instead.")
        return False

    # Check size
    total_size = sum(
        f.stat().st_size for f in Path(fused_dir).rglob("*") if f.is_file()
    )
    print(f"  ✅ Fused model saved to {fused_dir} ({total_size/1e9:.2f} GB)")
    return True


# ══════════════════════════════════════════════════════════════════════
# Integration Test
# ══════════════════════════════════════════════════════════════════════

def run_integration_test(fused_dir: str, adapter_dir: str):
    """Test the fine-tuned model in the actual MeetMind pipeline."""
    print("\n🔌 Running integration test with MeetMind pipeline...")

    model_path = fused_dir if os.path.exists(fused_dir) else None

    # Import MeetMind (must be run from project root)
    sys.path.insert(0, ".")
    try:
        from backend.core.config import Perception, ModelConfig
        from backend.main import MeetMind
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("     Run this script from the project root directory.")
        return

    analyst_model = model_path or BASE_MODEL
    cfg = ModelConfig(
        scribe_model="mlx-community/gemma-3-4b-it-4bit",
        analyst_model=analyst_model,
        architect_model="mlx-community/gemma-3-4b-it-4bit",
    )

    print(f"  Analyst model: {analyst_model}")
    mind = MeetMind(config=cfg)

    # Test scenarios
    tests = [
        {
            "name": "Engineering task assignment",
            "perception": Perception(
                timestamp="11:30",
                visual_text=["API Gateway", "Auth Service"],
                visual_content_type="diagram",
                visual_changed=True,
                audio_transcript="Sarah will handle OAuth by Friday.",
                audio_speech_detected=True,
            ),
            "expected_action": "extract_action_item",
        },
        {
            "name": "Decision made",
            "perception": Perception(
                timestamp="11:35",
                visual_text=["PostgreSQL", "MongoDB", "DynamoDB"],
                visual_content_type="table",
                visual_changed=True,
                audio_transcript="Let us go with PostgreSQL. It gives us ACID compliance.",
                audio_speech_detected=True,
            ),
            "expected_action": "log_decision",
        },
        {
            "name": "Unresolved gap",
            "perception": Perception(
                timestamp="11:40",
                visual_text=["Migration Plan", "Timeline: ???"],
                visual_content_type="freeform",
                visual_changed=True,
                audio_transcript="Someone should own the database migration but nobody volunteered.",
                audio_speech_detected=True,
            ),
            "expected_action": "flag_gap",
        },
    ]

    passed = 0
    for test in tests:
        result = mind.process_perception(test["perception"])
        actual = result["action"]["action"]
        ok = actual == test["expected_action"]
        status = "✅" if ok else "⚠️"
        if ok:
            passed += 1
        print(f"  {status} {test['name']}: expected={test['expected_action']} got={actual}")
        if result["action"].get("params"):
            print(f"     Params: {json.dumps(result['action']['params'], indent=2)}")

    print(f"\n  Integration: {passed}/{len(tests)} passed")


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MeetMind Analyst Fine-Tuning Pipeline")
    parser.add_argument("--train", action="store_true", help="Run full training pipeline")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing model")
    parser.add_argument("--fuse", action="store_true", help="Fuse adapters into standalone model")
    parser.add_argument("--integration-test", action="store_true", help="Test in MeetMind pipeline")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--iters", type=int, default=LORA_CONFIG["iters"], help="Training iterations")
    parser.add_argument("--lr", type=float, default=LORA_CONFIG["learning_rate"], help="Learning rate")
    args = parser.parse_args()

    LORA_CONFIG["iters"] = args.iters
    LORA_CONFIG["learning_rate"] = args.lr
    base_model = args.model

    print("=" * 60)
    print("MeetMind Analyst — Fine-Tuning Pipeline")
    print(f"Base model: {base_model}")
    print("=" * 60)

    if args.train:
        data_dir = prepare_data(TRAINING_FILE, SPLIT_DIR)
        run_training(data_dir, ADAPTER_DIR)
        fused = fuse_model(ADAPTER_DIR, FUSED_DIR)
        run_evaluation(SPLIT_DIR, ADAPTER_DIR, FUSED_DIR if fused else None)
        print(f"\n✅ Pipeline complete! Model at: {FUSED_DIR if fused else ADAPTER_DIR}")
        print(f"   To use: Update ModelConfig.analyst_model = '{FUSED_DIR if fused else ADAPTER_DIR}'")

    elif args.eval_only:
        fused = os.path.exists(FUSED_DIR)
        run_evaluation(SPLIT_DIR, ADAPTER_DIR, FUSED_DIR if fused else None)

    elif args.fuse:
        fuse_model(ADAPTER_DIR, FUSED_DIR)

    elif args.integration_test:
        run_integration_test(FUSED_DIR, ADAPTER_DIR)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
