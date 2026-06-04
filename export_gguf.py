# export_gguf.py
import os
from unsloth import FastModel
from peft import PeftModel

MODEL_ID = "./models/gemma-4-e4b-it"
ADAPTER_DIR = "./outputs/gemma4-e4b-ha"
GGUF_DIR = "./outputs/gemma4-e4b-ha-gguf"

print("Loading base model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    load_in_4bit=False,  # need full precision for merging
    full_finetuning=False,
    use_exact_model_name=True,
)

print("Merging LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()

print("Exporting to GGUF (q4_k_s)...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_s",
)
print(f"Done! GGUF saved to {GGUF_DIR}/")
