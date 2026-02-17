import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def final_compatibility_fix():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell_index = 2 # The main logic cell
    source = nb['cells'][cell_index]['source']
    
    new_source = []
    skip_until_next_marker = False
    
    for line in source:
        # Revert fp16 to False to avoid GradScaler/BFloat16 conflict on T4
        if "\"fp16\": True," in line:
            new_source.append("    \"fp16\": False,                     # ✅ REVERT: Evita conflicto BFloat16 en T4\n")
            continue
            
        # Ensure we keep the evaluation batch size fixes
        if "\"per_device_eval_batch_size\": 1," in line:
            new_source.append(line)
            continue
            
        new_source.append(line)

    # Also, let's make sure we clean up any old "Nuclear" or "Ultimate" fix blocks that might be messy
    # and just keep a clean casting to float16 for safety.
    
    clean_source = []
    in_fix_block = False
    for line in new_source:
        if "# 🔥 NUCLEAR FIX" in line or "# 🔥 ULTIMATE FIX" in line:
            in_fix_block = True
            clean_source.append("# 🔥 SAFE CAST TO FLOAT16 (Evita BFloat16)\n")
            clean_source.append("for p in model.parameters():\n")
            clean_source.append("    if p.dtype == torch.bfloat16:\n")
            clean_source.append("        p.data = p.data.to(torch.float16)\n")
            continue
        
        if in_fix_block:
            if line.startswith("# ==================") or line.startswith("print(\"🔍 Buscando"):
                in_fix_block = False
                clean_source.append(line)
            continue
            
        clean_source.append(line)

    nb['cells'][cell_index]['source'] = clean_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated: fp16 set to False, OOM fixes preserved.")

if __name__ == "__main__":
    final_compatibility_fix()
