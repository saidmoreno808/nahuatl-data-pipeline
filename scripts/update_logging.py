import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def update_logging_steps():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Looking for the training config in Cell 2
    cell2 = nb['cells'][2]
    source = cell2['source']
    
    new_source = []
    for line in source:
        if "\"logging_steps\": 10," in line:
            new_source.append("    \"logging_steps\": 1, \n")
            new_source.append("    \"logging_first_step\": True, \n")
            continue
        new_source.append(line)
        
    cell2['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated: logging_steps set to 1.")

if __name__ == "__main__":
    update_logging_steps()
