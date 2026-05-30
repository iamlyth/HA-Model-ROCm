"""
Training script for fine-tuning unsloth/gemma-4-E4B-it on the
acon96/Home-Assistant-Requests-V2 dataset using Unsloth + LoRA.

Usage:
    Test run (1 epoch over 1000 rows):  python train.py --test
    Full run (209k rows):  python train.py

To cache the base model locally (HuggingFace-independent):
    huggingface-cli download unsloth/gemma-4-E4B-it --local-dir ./models/gemma-4-E4B-it
"""

import os
import argparse
from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────

HF_MODEL_ID   = "unsloth/gemma-4-E4B-it"
LOCAL_MODEL   = "./models/gemma-4-e4b-it"
OUTPUT_DIR    = "outputs/gemma4-e4b-ha"
GGUF_DIR      = "outputs/gemma4-e4b-ha-gguf"

# LoRA settings
LORA_RANK     = 8
LORA_ALPHA    = 8

# Training settings
MAX_SEQ_LENGTH = 2048
BATCH_SIZE     = 8
GRAD_ACCUM     = 4        # effective batch size = 4
LEARNING_RATE  = 2e-4
WARMUP_STEPS   = 5
EPOCHS_FULL    = 1        # used in full mode

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run a quick test with 100 steps")
args = parser.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_ID = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else HF_MODEL_ID
print(f"Loading model from: {MODEL_ID}")

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
    use_exact_model_name=True,  # prevent remapping to bnb-4bit variant
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # text only
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ── Chat template ─────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

print("Loading dataset: acon96/Home-Assistant-Requests-V2")
dataset = load_dataset("acon96/Home-Assistant-Requests-V2", split="train", cache_dir="./datasets")

if args.test:
    print("Test mode: using 1000 rows")
    dataset = dataset.select(range(1000))
else:
    print(f"Full mode: using {len(dataset)} rows")

def format_example(example):
    try:
        messages = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            tool_calls = msg.get("tool_calls")

            # Normalize content from list to string
            if isinstance(content, list):
                content = " ".join(c["text"] for c in content if c.get("type") == "text")

            if role == "system":
                messages.append({"role": "system", "content": content or ""})
            elif role == "tool":
                # Parse tool result from JSON
                try:
                    import json
                    result = json.loads(content)
                    content = result.get("tool_result", content)
                except Exception:
                    pass
                messages.append({"role": "tool", "content": str(content)})
            elif role == "assistant" and tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                })
            else:
                messages.append({"role": role, "content": content or ""})

        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tools=example.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
        }
    except Exception as e:
        return {"text": ""}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

# ── Trainer ───────────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=-1,                          # always let epochs control
    num_train_epochs=1,                    # always 1 epoch
    fp16=False,
    bf16=True,
    logging_steps=1,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    resume_from_checkpoint=True,
)

# ── Train ─────────────────────────────────────────────────────────────────────

print("Starting training...")
trainer.train()
print("Training complete.")

# ── Save ──────────────────────────────────────────────────────────────────────

print("Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if not args.test:
    print("Exporting to GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(
        GGUF_DIR,
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF saved to: {GGUF_DIR}/")

print("Done!")
