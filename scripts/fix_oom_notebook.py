import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def fix_oom_in_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the training cell (Cell 2)
    cell2 = nb['cells'][2]
    source = cell2['source']
    
    new_source = []
    found_sft_config = False
    
    for line in source:
        # Update per_device_train_batch_size to ensure eval batch size is also handled
        if "\"per_device_train_batch_size\": 1," in line:
            new_source.append(line)
            new_source.append("    \"per_device_eval_batch_size\": 1, \n")
            new_source.append("    \"eval_accumulation_steps\": 4, \n")
            continue
            
        # Fix fp16 setting
        if "\"fp16\": False," in line:
            new_source.append("    \"fp16\": True,\n")
            continue
            
        new_source.append(line)
        
    cell2['source'] = new_source
    print("Optimization: Set eval_batch_size=1 and fp16=True.")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated with OOM fixes.")

if __name__ == "__main__":
    fix_oom_in_notebook()
