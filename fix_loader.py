"""
Removes leftover debug patches from unsloth's loader.py.
Run after installing unsloth if you see SyntaxError in loader.py.
"""
import os

path = ".venv/lib/python3.12/site-packages/unsloth/models/loader.py"
if not os.path.exists(path):
    print("  loader.py not found, skipping")
    exit(0)

with open(path) as f:
    content = f.read()

# Known debug patches to remove
patches = [
    (
        '                "Please separate the LoRA and base models to 2 repos."\n'
        '        print("PRE-CALL model_config:", model_config, "peft_config:", peft_config)\n'
        '            )',
        '                "Please separate the LoRA and base models to 2 repos."\n'
        '            )'
    ),
    (
        '            print("AUTOCONFIG2 ERROR:", str(error))\n'
        '            autoconfig_error',
        '            autoconfig_error'
    ),
    (
        '            print("DEBUG2 peft_config:", peft_config)\n'
        '        print("DEBUG2 model_config:", model_config)\n',
        ''
    ),
]

fixed = False
for old, new in patches:
    if old in content:
        content = content.replace(old, new)
        fixed = True

if fixed:
    with open(path, 'w') as f:
        f.write(content)
    print("  Fixed debug patches in loader.py")
else:
    print("  loader.py is clean")
