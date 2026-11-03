# export_gguf.py
import os
from unsloth import FastModel
from peft import PeftModel

MODEL_ID = "./models/nemotron-e4b"
ADAPTER_DIR = "./outputs/nemotron-e4b-ha"
GGUF_DIR = "./outputs/nemotron-e4b-ha-gguf"

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

print("Exporting to GGUF (Q4_K_M)...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"Done! GGUF saved to {GGUF_DIR}/")
