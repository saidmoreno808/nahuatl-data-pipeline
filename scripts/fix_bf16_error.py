import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def fix_bfloat16_error():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # We need to update the casting logic in Cell 2 (index 1 in the notebook array if it's the 2nd cell)
    # Based on previous view_file, Cell 2 (index 1) has the model loading.
    # Actually, the views show Cell 1 (index 0) is markdown, Cell 2 (index 1) is installation, 
    # Cell 3 (index 2) is the main logic.
    
    cell_index = 2
    source = nb['cells'][cell_index]['source']
    
    new_source = []
    found_cast = False
    for line in source:
        # Replace the old casting loop with a more aggressive one
        if "# Manual Cast a Float16" in line:
            new_source.append("# 🔥 FIX COMPATIBILIDAD BFloat16 (Evita NotImplementedError)\n")
            new_source.append("print(\"🛡️ Asegurando que no existan tensores BFloat16...\")\n")
            new_source.append("for name, param in model.named_parameters():\n")
            new_source.append("    if param.dtype == torch.bfloat16:\n")
            new_source.append("        param.data = param.data.to(torch.float16)\n")
            new_source.append("\n")
            new_source.append("# Cast trainable to float32 if needed for stability with GradScaler\n")
            new_source.append("for name, param in model.named_parameters():\n")
            new_source.append("    if param.requires_grad:\n")
            new_source.append("        param.data = param.data.to(torch.float32)\n")
            found_cast = True
            continue
        
        # Skip the next 3 lines if we just added the new cast
        if found_cast and ("for name, param" in line or "param.data.to(torch.float16)" in line or "requires_grad" in line):
            continue
            
        new_source.append(line)

    nb['cells'][cell_index]['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated with robust casting to avoid BFloat16 error.")

if __name__ == "__main__":
    fix_bfloat16_error()
