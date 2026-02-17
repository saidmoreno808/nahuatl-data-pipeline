import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def apply_nuclear_bf16_fix():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell_index = 2 # The main logic cell
    source = nb['cells'][cell_index]['source']
    
    new_source = []
    skip_next = 0
    
    # We will looking for the model loading part to insert a better fix
    fixed_logic = [
        "\n",
        "# 🔥 NUCLEAR FIX FOR BFLOAT16 (NotImplementedError on T4)\n",
        "print(\"🧹 Detectando y eliminando BFloat16...\")\n",
        "found_bf16 = False\n",
        "for name, param in model.named_parameters():\n",
        "    if param.dtype == torch.bfloat16:\n",
        "        print(f\"  - Casting param {name} to float16\")\n",
        "        param.data = param.data.to(torch.float16)\n",
        "        found_bf16 = True\n",
        "\n",
        "for name, buf in model.named_buffers():\n",
        "    if buf.dtype == torch.bfloat16:\n",
        "        print(f\"  - Casting buffer {name} to float16\")\n",
        "        buf.data = buf.data.to(torch.float16)\n",
        "        found_bf16 = True\n",
        "\n",
        "if not found_bf16:\n",
        "    print(\"✅ No se detectaron tensores BFloat16.\")\n",
        "\n",
        "# Asegurar dtypes en módulos para evitar que TRL los cambie\n",
        "for m in model.modules():\n",
        "    if hasattr(m, \"dtype\") and m.dtype == torch.bfloat16:\n",
        "        m.to(torch.float16)\n",
        "\n",
        "model.config.torch_dtype = torch.float16\n",
        "model.config.use_cache = False\n",
        "\n"
    ]
    
    i = 0
    while i < len(source):
        line = source[i]
        
        # Remove old fix attempts
        if "# 🔥 ULTIMATE FIX FOR BFLOAT16" in line or "# 🔥 FIX COMPATIBILIDAD BFloat16" in line:
            # Found an old block, skip it until the next major section or empty line
            while i < len(source) and not source[i].startswith("# ==================") and source[i].strip() != "":
                i += 1
            continue
            
        if "model = get_peft_model(model, peft_config)" in line:
            new_source.append(line)
            new_source.extend(fixed_logic)
            i += 1
            continue
            
        new_source.append(line)
        i += 1

    nb['cells'][cell_index]['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated with Nuclear BFloat16 fix.")

if __name__ == "__main__":
    apply_nuclear_bf16_fix()
