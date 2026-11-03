

## To Train

This fine-tunes **NVIDIA Nemotron 3.5 Bolt E4B** (GRPO post-trained Gemma 4 E4B)
on the Home-Assistant-Requests-V2 dataset. Nemotron is already optimized for
tool use and instruction following — a much better starting point than base
Gemma 4 E4B-it for HA intent recognition.

### Download model into cache
```
uv run huggingface-cli download nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 --local-dir ./models/nemotron-e4b
```

### Run training
```
# Test run (1000 rows)
uv run python train.py --test

# Full run (209k rows)
uv run python train.py
```

### Export GGUF (standalone, if training already completed)
```
uv run python export_gguf.py
```

The exported Q4_K_M GGUF is optimized for the Jetson Orin Nano 8GB:
- mmap keeps PLE embeddings evicted under memory pressure (~3GB active)
- Deploy to `/home/lalobied/models/ha-gemma4-e4b.gguf` on the Jetson

