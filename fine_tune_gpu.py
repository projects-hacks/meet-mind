"""GPU-optimized fine-tuning for MeetMind Analyst agent.

Runs on NVIDIA RTX PRO 6000 Blackwell (98GB VRAM) using PyTorch + HuggingFace.
Uses LoRA via PEFT for efficient fine-tuning of Gemma 3.

Usage:
    python3 fine_tune_gpu.py                    # Full pipeline: prepare → train → eval
    python3 fine_tune_gpu.py --eval-only        # Just evaluate existing model
    python3 fine_tune_gpu.py --model google/gemma-3-4b-it  # Use different model
"""

import argparse
import json
import os
import random
import re
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "google/gemma-3-4b-it"
TRAINING_FILE = "data/analyst_training.jsonl"
OUTPUT_DIR = "outputs/gpu-analyst-lora"
FUSED_DIR = "outputs/gpu-analyst-fused"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EVAL_SPLIT = 0.15  # 15% for eval
SEED = 42


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_and_split_data(training_file: str):
    """Load JSONL training data and split into train/eval with stratification."""
    print(f"\n📂 Loading data from {training_file}")

    with open(training_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"   Loaded {len(examples)} examples")

    # Group by action for stratified split
    by_action = {}
    for ex in examples:
        action = json.loads(ex["messages"][2]["content"])["action"]
        by_action.setdefault(action, []).append(ex)

    train_examples = []
    eval_examples = []

    random.seed(SEED)

    print("\n   Distribution:")
    for action, items in sorted(by_action.items()):
        random.shuffle(items)
        n_eval = max(2, int(len(items) * EVAL_SPLIT))  # At least 2 eval per class
        eval_examples.extend(items[:n_eval])
        train_examples.extend(items[n_eval:])
        print(f"     {action}: {len(items)} total → {len(items) - n_eval} train / {n_eval} eval")

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    print(f"\n   Split: {len(train_examples)} train / {len(eval_examples)} eval")
    return train_examples, eval_examples


def format_for_training(examples: list, tokenizer) -> Dataset:
    """Convert message triples into formatted text for SFT training.

    Uses Gemma's chat template to format system + user + assistant messages
    into the expected token format.
    """
    texts = []
    for ex in examples:
        messages = ex["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception:
            # Fallback: manual formatting
            text = f"<start_of_turn>system\n{messages[0]['content']}<end_of_turn>\n"
            text += f"<start_of_turn>user\n{messages[1]['content']}<end_of_turn>\n"
            text += f"<start_of_turn>model\n{messages[2]['content']}<end_of_turn>"
            texts.append(text)

    return Dataset.from_dict({"text": texts})


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train(model_name: str, training_file: str, output_dir: str):
    """Run LoRA fine-tuning on GPU."""
    print(f"\n🚀 Starting GPU fine-tuning")
    print(f"   Model: {model_name}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

    start = time.time()

    # ── Load data ──
    train_data, eval_data = load_and_split_data(training_file)

    # ── Load tokenizer ──
    print(f"\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Format data ──
    train_dataset = format_for_training(train_data, tokenizer)
    eval_dataset = format_for_training(eval_data, tokenizer)
    print(f"   Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # ── Load model ──
    print(f"\n📦 Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── Apply LoRA ──
    print(f"\n🔧 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args (SFTConfig extends TrainingArguments with SFT-specific params) ──
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=MAX_SEQ_LENGTH,
    )

    # ── Custom SFTTrainer subclass for Gemma 3 multimodal ──
    # Gemma 3 4B is vision+text. token_type_ids tells the model:
    #   0 = text token, 1 = image token
    # We inject token_type_ids=0 for all tokens at compute_loss time.
    class Gemma3SFTTrainer(SFTTrainer):
        """SFTTrainer that injects token_type_ids=0 (text-only) for Gemma 3 multimodal."""
        def compute_loss(self, model, inputs, *args, **kwargs):
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            return super().compute_loss(model, inputs, *args, **kwargs)

    # ── Trainer ──
    trainer = Gemma3SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train! ──
    print(f"\n🏋️ Training...")
    trainer.train()

    # ── Save ──
    print(f"\n💾 Saving adapters to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start
    print(f"\n✅ Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════
# Fusion
# ══════════════════════════════════════════════════════════════════════

def fuse_model(model_name: str, adapter_dir: str, fused_dir: str):
    """Merge LoRA adapters into base model for fast inference."""
    print(f"\n🔀 Fusing LoRA adapters into base model...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    print(f"   Saving fused model to {fused_dir}")
    model.save_pretrained(fused_dir)
    tokenizer.save_pretrained(fused_dir)

    # Report size
    total_bytes = sum(
        os.path.getsize(os.path.join(fused_dir, f))
        for f in os.listdir(fused_dir)
        if f.endswith(".safetensors")
    )
    print(f"   Fused model size: {total_bytes / (1024**3):.2f} GB")
    print(f"✅ Fusion complete")


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def extract_action(text: str):
    """Extract the action name from model output, handling varied formats."""
    # Try JSON first
    try:
        data = json.loads(text.strip())
        return data.get("action")
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try regex patterns
    patterns = [
        r'"action"\s*:\s*"([^"]+)"',
        r"'action'\s*:\s*'([^']+)'",
        r"action[=:]\s*(\w+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    return None


def evaluate(model_name_or_path: str, training_file: str, is_fused: bool = False):
    """Evaluate model accuracy on held-out test set."""
    print(f"\n📊 Evaluating model: {model_name_or_path}")

    # Load data
    _, eval_data = load_and_split_data(training_file)

    # Load model
    print(f"\n📦 Loading model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_fused:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # LoRA adapter
        base_name = DEFAULT_MODEL
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path)

    model.eval()

    # Run evaluation
    correct = 0
    total = 0
    per_action = {}
    results = []

    print(f"\n🧪 Evaluating {len(eval_data)} examples...")

    for i, ex in enumerate(eval_data):
        messages = ex["messages"]
        expected_action = json.loads(messages[2]["content"])["action"]

        # Build prompt (system + user, no assistant)
        prompt_messages = messages[:2]  # system + user only
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"<start_of_turn>system\n{messages[0]['content']}<end_of_turn>\n"
            prompt += f"<start_of_turn>user\n{messages[1]['content']}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted_action = extract_action(response)

        is_correct = predicted_action == expected_action
        correct += int(is_correct)
        total += 1

        # Track per-action
        if expected_action not in per_action:
            per_action[expected_action] = {"correct": 0, "total": 0}
        per_action[expected_action]["total"] += 1
        per_action[expected_action]["correct"] += int(is_correct)

        emoji = "✅" if is_correct else "❌"
        print(f"  [{i+1}/{len(eval_data)}] {emoji} Expected: {expected_action}, Got: {predicted_action}")

        results.append({
            "expected": expected_action,
            "predicted": predicted_action,
            "correct": is_correct,
            "response_preview": response[:200],
        })

    # Report
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*60}")

    print(f"\n  Per-action breakdown:")
    for action in sorted(per_action.keys()):
        stats = per_action[action]
        pct = stats["correct"] / stats["total"] * 100
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        emoji = "✅" if pct == 100 else "⚠️" if pct >= 50 else "❌"
        print(f"    {emoji} {action}: {stats['correct']}/{stats['total']} ({pct:.0f}%) {bar}")

    # Save results
    results_file = os.path.join(os.path.dirname(model_name_or_path), "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_action": per_action,
            "results": results,
        }, f, indent=2)
    print(f"\n📄 Results saved to {results_file}")

    return accuracy


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPU Fine-Tuning for MeetMind Analyst")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name/path")
    parser.add_argument("--data", default=TRAINING_FILE, help="Training data JSONL")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output dir for adapters")
    parser.add_argument("--fused-dir", default=FUSED_DIR, help="Output dir for fused model")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate")
    parser.add_argument("--no-fuse", action="store_true", help="Skip fusion step")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    if args.eval_only:
        # Evaluate fused model if exists, else adapter
        if os.path.exists(args.fused_dir):
            evaluate(args.fused_dir, args.data, is_fused=True)
        elif os.path.exists(args.output):
            evaluate(args.output, args.data, is_fused=False)
        else:
            print("❌ No model found to evaluate")
            sys.exit(1)
        return

    # Full pipeline
    print("=" * 60)
    print("  MeetMind Analyst — GPU Fine-Tuning Pipeline")
    print("=" * 60)

    # 1. Train
    model, tokenizer = train(args.model, args.data, args.output)
    del model
    torch.cuda.empty_cache()

    # 2. Fuse
    if not args.no_fuse:
        fuse_model(args.model, args.output, args.fused_dir)

    # 3. Evaluate
    eval_path = args.fused_dir if not args.no_fuse else args.output
    is_fused = not args.no_fuse
    evaluate(eval_path, args.data, is_fused=is_fused)

    print(f"\n🎉 Pipeline complete!")
    print(f"   Fused model: {args.fused_dir}")
    print(f"   Ready to copy back to Mac for deployment")


if __name__ == "__main__":
    main()
