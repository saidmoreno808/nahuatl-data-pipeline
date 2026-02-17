import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def fix_bf16_persistence():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell_index = 2 # The main logic cell
    source = nb['cells'][cell_index]['source']
    
    new_source = []
    skip_next = 0
    for i, line in enumerate(source):
        if skip_next > 0:
            skip_next -= 1
            continue
            
        # Look for the previous fix markers or the peft model line to insert the fix right after it
        if "model = get_peft_model(model, peft_config)" in line:
            new_source.append(line)
            new_source.append("\n")
            new_source.append("# 🔥 ULTIMATE FIX FOR BFLOAT16 VS FP16 (NotImplementedError)\n")
            new_source.append("print(\"🧹 Realizando limpieza agresiva de BFloat16...\")\n")
            new_source.append("# 1. Convertir TODOS los parámetros y buffers no-entrenables a float16\n")
            new_source.append("for p in model.parameters():\n")
            new_source.append("    if not p.requires_grad:\n")
            new_source.append("        p.data = p.data.to(torch.float16)\n")
            new_source.append("\n")
            new_source.append("# 2. Asegurar que los adaptadores LoRA (entrenables) estén en float32 para el GradScaler\n")
            new_source.append("for p in model.parameters():\n")
            new_source.append("    if p.requires_grad:\n")
            new_source.append("        p.data = p.data.to(torch.float32)\n")
            new_source.append("\n")
            new_source.append("# 3. Forzar el dtype en la configuración del modelo\n")
            new_source.append("model.config.torch_dtype = torch.float16\n")
            continue
            
        # Remove any existing manual cast blocks to avoid mess
        if "# 🔥 FIX COMPATIBILIDAD BFloat16" in line or "# Manual Cast a Float16" in line:
            # Skip this block (around 10 lines)
            j = i
            while j < len(source) and not source[j].startswith("# =================="):
                j += 1
            skip_next = j - i - 1
            continue
            
        new_source.append(line)

    nb['cells'][cell_index]['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated with ultimate BFloat16 fix.")

if __name__ == "__main__":
    fix_bf16_persistence()
