#!/usr/bin/env bash
set -e

echo "=== HAModel Setup for NixOS + gfx1151 ==="

# ── Step 1: Install torch from gfx1151 nightly (with deps) ────────────────────
# Must be done FIRST and separately — other packages pull CUDA torch otherwise.
echo ""
echo ">>> Installing torch + ROCm SDK (gfx1151 native)..."
uv pip install "torch==2.10.0+rocm7.13.0a20260513" \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --index-strategy unsafe-best-match \
  --force-reinstall

# ── Step 2: Install torchvision and triton from gfx1151 (no deps) ─────────────
echo ""
echo ">>> Installing torchvision + triton (gfx1151 native, no deps)..."
uv pip install \
  "torchvision==0.25.0+rocm7.13.0a20260513" \
  "triton==3.7.0+git18f89f64.rocm7.13.0a20260411" \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --no-deps \
  --force-reinstall

# ── Step 3: Install all other dependencies (pinned, no deps) ──────────────────
echo ""
echo ">>> Installing pinned dependencies (no deps to avoid CUDA torch)..."
uv pip install \
  "accelerate==1.13.0" \
  "annotated-doc==0.0.4" \
  "anyio==4.13.0" \
  "bitsandbytes==0.49.2" \
  "certifi==2026.5.20" \
  "chardet==7.4.3" \
  "charset-normalizer==3.4.7" \
  "click==8.4.1" \
  "datasets==4.8.5" \
  "dill==0.4.1" \
  "filelock==3.29.0" \
  "fsspec==2026.3.0" \
  "h11==0.16.0" \
  "hf-transfer==0.1.9" \
  "hf-xet==1.5.0" \
  "httpcore==1.0.9" \
  "httpx==0.28.1" \
  "huggingface-hub==1.17.0" \
  "idna==3.17" \
  "jinja2==3.1.6" \
  "markdown-it-py==4.2.0" \
  "markupsafe==3.0.3" \
  "mdurl==0.1.2" \
  "mpmath==1.3.0" \
  "multiprocess==0.70.19" \
  "nest-asyncio==1.6.0" \
  "networkx==3.6.1" \
  "numpy==2.4.6" \
  "packaging==26.2" \
  "pandas==3.0.3" \
  "peft==0.19.1" \
  "pillow==12.2.0" \
  "protobuf==7.35.0" \
  "psutil==7.2.2" \
  "pyarrow==24.0.0" \
  "pydantic==2.13.4" \
  "pygments==2.20.0" \
  "python-dateutil==2.9.0.post0" \
  "pyyaml==6.0.3" \
  "regex==2026.5.9" \
  "requests==2.34.2" \
  "rich==15.0.0" \
  "safetensors==0.7.0" \
  "sentencepiece==0.2.1" \
  "setuptools==82.0.1" \
  "shellingham==1.5.4" \
  "six==1.17.0" \
  "sympy==1.14.0" \
  "tokenizers==0.22.2" \
  "torchao==0.17.0" \
  "tqdm==4.67.3" \
  "transformers==5.5.0" \
  "trl==1.5.1" \
  "typer==0.25.1" \
  "typing-extensions==4.15.0" \
  "tyro==1.0.13" \
  "urllib3==2.7.0" \
  "xxhash==3.7.0" \
  "attrs==26.1.0" \
  "attr==0.3.2" \
  "attrs==26.1.0" \
  "attrs==26.1.0" \
  "attrs==26.1.0" \
  --no-deps \
  --index-strategy unsafe-best-match

# ── Step 4: Install unsloth and unsloth-zoo (no deps) ─────────────────────────
echo ""
echo ">>> Installing unsloth (no deps)..."
uv pip install \
  "unsloth==2026.5.8" \
  "unsloth-zoo==2026.5.4" \
  --no-deps \
  --index-strategy unsafe-best-match

# ── Step 5: Fix leftover debug patches in unsloth ─────────────────────────────
echo ""
echo ">>> Cleaning up any debug patches in unsloth..."
python3 fix_loader.py 2>/dev/null || true

# ── Step 6: Fix rocm_sdk _dist_info ───────────────────────────────────────────
echo ""
echo ">>> Patching rocm_sdk library registry..."
python3 - << 'PYEOF'
import os

path = ".venv/lib/python3.12/site-packages/rocm_sdk/_dist_info.py"
if not os.path.exists(path):
    print("  rocm_sdk/_dist_info.py not found, skipping")
    exit(0)

with open(path) as f:
    content = f.read()

additions = "\n# Added by setup.sh for gfx1151 compatibility\n"
entries = [
    ('amd_comgr',            'core',      'libamd_comgr.so.3'),
    ('rocprofiler-sdk',      'core',      'librocprofiler-sdk.so.1'),
    ('roctracer64',          'core',      'libroctracer64.so.4'),
    ('rocm_sysdeps_liblzma', 'core',      'libroctracer64.so.4'),
    ('rocm-openblas',        'devel',     'libopenblas.so.0'),
    ('rocm_smi64',           'core',      'librocm_smi64.so.7'),
    ('hipsparselt',          'libraries', 'libhipsparselt.so.0'),
    ('hipdnn',               'libraries', 'libMIOpen.so.1'),
]

for shortname, pkg, soname in entries:
    if f'"{shortname}"' not in content:
        additions += f'LibraryEntry("{shortname}", "{pkg}", "{soname}")\n'

if additions.strip() != "# Added by setup.sh for gfx1151 compatibility":
    content += additions
    with open(path, 'w') as f:
        f.write(content)
    print("  Patched rocm_sdk/_dist_info.py")
else:
    print("  rocm_sdk/_dist_info.py already patched")
PYEOF

# ── Step 7: Fix torch _rocm_init.py ───────────────────────────────────────────
echo ""
echo ">>> Patching torch ROCm init..."
python3 - << 'PYEOF'
path = ".venv/lib/python3.12/site-packages/torch/_rocm_init.py"
expected = '''def initialize():
    try:
        import rocm_sdk
        rocm_sdk.initialize_process(
            preload_shortnames=["amd_comgr", "amdhip64", "rocprofiler-sdk", "rocprofiler-sdk-roctx", "roctracer64", "roctx64", "hiprtc", "hipblas", "hipfft", "hiprand", "hipsparse", "hipsparselt", "hipsolver", "rccl", "hipblaslt", "miopen"],
            check_version="*")
    except Exception as e:
        print(f"rocm_sdk init warning: {e}")
'''
with open(path, 'w') as f:
    f.write(expected)
print("  Patched torch/_rocm_init.py")
PYEOF

# ── Step 8: Fix triton build.py Python.h path ─────────────────────────────────
echo ""
echo ">>> Patching triton build.py for NixOS Python headers..."
python3 - << 'PYEOF'
import subprocess, os

result = subprocess.run(
    ["find", "/nix/store", "-name", "Python.h", "-path", "*3.12*"],
    capture_output=True, text=True
)
paths = [p for p in result.stdout.strip().split('\n') if p and 'source' not in p]
if not paths:
    print("  WARNING: Python.h not found in nix store")
    exit(0)

py_h = paths[0]
include_dir = os.path.dirname(py_h)
print(f"  Found Python.h at: {py_h}")

path = ".venv/lib/python3.12/site-packages/triton/runtime/build.py"
with open(path) as f:
    content = f.read()

old = '    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]'
new = f'    py_include_dir = "{include_dir}"'

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print("  Patched triton/runtime/build.py")
elif include_dir in content:
    print("  triton/runtime/build.py already patched")
else:
    print("  WARNING: Could not find patch target in triton/runtime/build.py")
PYEOF

# ── Step 9: Fix bitsandbytes ROCm symlink ─────────────────────────────────────
echo ""
echo ">>> Patching bitsandbytes ROCm library..."
python3 - << 'PYEOF'
import os, glob

bnb_dir = ".venv/lib/python3.12/site-packages/bitsandbytes"
if not os.path.exists(bnb_dir):
    print("  bitsandbytes not found, skipping")
    exit(0)

rocm_libs = sorted(glob.glob(f"{bnb_dir}/libbitsandbytes_rocm*.so"))
# Filter out the rocm83 symlink itself
rocm_libs = [l for l in rocm_libs if 'rocm83' not in l]
if not rocm_libs:
    print("  No ROCm bitsandbytes libs found")
    exit(0)

latest = rocm_libs[-1]
print(f"  Found: {os.path.basename(latest)}")

target = f"{bnb_dir}/libbitsandbytes_rocm83.so"
if os.path.exists(target) or os.path.islink(target):
    os.remove(target)
os.symlink(os.path.basename(latest), target)
print(f"  Created symlink: libbitsandbytes_rocm83.so -> {os.path.basename(latest)}")
PYEOF

# ── Step 10: Fix rocm_sdk_core soname symlinks ─────────────────────────────────
echo ""
echo ">>> Creating ROCm library version symlinks..."
python3 - << 'PYEOF'
import os

lib_dir = ".venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"
if not os.path.exists(lib_dir):
    print("  _rocm_sdk_core/lib not found, skipping")
    exit(0)

symlinks = [
    ("librocm_smi64.so.7", "librocm_smi64.so.1"),
    ("libamdhip64.so.6",   "libamdhip64.so.7"),
    ("libhiprtc.so.6",     "libhiprtc.so.7"),
]

for src, dst in symlinks:
    src_path = os.path.join(lib_dir, src)
    dst_path = os.path.join(lib_dir, dst)
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        os.symlink(src, dst_path)
        print(f"  Created: {dst} -> {src}")
    elif os.path.exists(dst_path):
        print(f"  Already exists: {dst}")
    else:
        print(f"  WARNING: source not found: {src}")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Verify with:"
echo "  nix-shell"
echo "  python -c 'import torch; print(torch.__version__); t = torch.tensor([1.0]).cuda(); print(t)'"
echo "  python -c 'from unsloth import FastModel; print(\"unsloth ok\")'"
echo ""
echo "Then start training:"
echo "  python train.py --test"
