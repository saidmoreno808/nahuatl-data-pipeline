import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def force_per_step_logging():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # The training config is in Cell 2
    cell2 = nb['cells'][2]
    source = cell2['source']
    
    new_source = []
    for line in source:
        # Replace logging_steps
        if "\"logging_steps\": 10," in line:
            new_source.append("    \"logging_steps\": 1, \n")
            new_source.append("    \"logging_first_step\": True, \n")
            continue
        new_source.append(line)
        
    cell2['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated: logging_steps=1, logging_first_step=True.")

if __name__ == "__main__":
    force_per_step_logging()
