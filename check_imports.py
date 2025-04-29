# check_imports.py
import importlib

modules_to_check = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "sympy",
    "alignment",
]

missing_modules = []

print("="*40)
print("🔎 Checking required Python modules...\n")

for module_name in modules_to_check:
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} found.")
    except ImportError as e:
        print(f"❌ {module_name} MISSING!")
        missing_modules.append(module_name)

print("\n" + "="*40)

if missing_modules:
    print(f"🚨 Missing modules: {', '.join(missing_modules)}")
    print("Please install them before running Slurm job.")
    exit(1)
else:
    print("🎉 All required modules are installed. You are good to go!")
    exit(0)
