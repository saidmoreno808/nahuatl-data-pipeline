import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def audit_and_fix_notebook():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Define common helpers (to be used in both training and eval parts if needed)
    # We'll put them in Cell 3 (the main training cell) or a separate cell.
    # Let's keep it in Cell 3 for simplicity as it was.
    
    # We need to find where the data loader ends and add the mapping logic.
    cell_index = 2 # The main training cell
    source = nb['cells'][cell_index]['source']
    
    # Check for mapping logic
    if not any("def formatting_prompts_func" in line for line in source):
        print("Restoring formatting_prompts_func and mapping logic...")
        
        mapping_code = [
            "\n",
            "# ==========================================\n",
            "# 2.5 PROMPT FORMATTING & PREPARATION\n",
            "# ==========================================\n",
            "alpaca_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
            "### Instruction:\n",
            "{}\n",
            "### Input:\n",
            "{}\n",
            "### Response:\n",
            "{}\"\"\"\n",
            "EOS_TOKEN = tokenizer.eos_token\n",
            "\n",
            "def formatting_prompts_func(examples):\n",
            "    if \"instruction\" in examples and \"input\" in examples and \"output\" in examples:\n",
            "        instructions, inputs, outputs = examples[\"instruction\"], examples[\"input\"], examples[\"output\"]\n",
            "    elif \"es\" in examples:\n",
            "         inputs = examples[\"es\"]\n",
            "         if \"nah\" in examples: outputs = examples[\"nah\"]\n",
            "         elif \"myn\" in examples: outputs = examples[\"myn\"]\n",
            "         else: outputs = [None] * len(inputs)\n",
            "         instructions = [\"Translate Spanish to Native Language\"] * len(inputs)\n",
            "    elif \"source\" in examples and \"target\" in examples:\n",
            "         inputs, outputs = examples[\"source\"], examples[\"target\"]\n",
            "         instructions = [\"Translate to Nahuatl\"] * len(inputs)\n",
            "    else:\n",
            "         col_list = list(examples.keys())\n",
            "         inputs, outputs = examples[col_list[0]], examples[col_list[1]]\n",
            "         instructions = [\"Process this text\"] * len(inputs)\n",
            "    \n",
            "    texts = []\n",
            "    for i, inp, out in zip(instructions, inputs, outputs):\n",
            "        if out is None or inp is None: \n",
            "            texts.append(None)\n",
            "            continue\n",
            "        text = alpaca_prompt.format(i, inp, out) + EOS_TOKEN\n",
            "        texts.append(text)\n",
            "    return { \"text\" : texts }\n",
            "\n",
            "cols = train_dataset.column_names\n",
            "print(f\"Applying prompt mapping to {len(train_dataset)} examples...\")\n",
            "train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=cols)\n",
            "eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=cols)\n",
            "\n",
            "train_dataset = train_dataset.filter(lambda x: x[\"text\"] is not None and len(x[\"text\"]) > 10)\n",
            "eval_dataset = eval_dataset.filter(lambda x: x[\"text\"] is not None and len(x[\"text\"]) > 10)\n",
            "\n",
            "def generate_translation(model, tokenizer, text):\n",
            "    # Helper needed for local evaluation\n",
            "    prompt = f\"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
            "\n",
            "### Instruction:\n",
            "Translate Spanish to Native Language\n",
            "\n",
            "### Input:\n",
            "{text}\n",
            "\n",
            "### Response:\n",
            "\"\"\"\n",
            "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)\n",
            "    with torch.no_grad():\n",
            "        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.3, do_sample=True)\n",
            "    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
            "    if \"### Response:\" in full_text: return full_text.split(\"### Response:\")[-1].strip()\n",
            "    return full_text.strip()\n",
            "\n"
        ]
        
        # We find where the dataset loading message is and insert after it.
        insertion_point = -1
        for i, line in enumerate(source):
            if "\"📊 Dataset cargado -> Train:" in line:
                insertion_point = i + 1
                break
        
        if insertion_point != -1:
            source = source[:insertion_point] + mapping_code + source[insertion_point:]
        else:
            # Fallback if marker not found
            source.extend(mapping_code)
            
        nb['cells'][cell_index]['source'] = source
        print("Restored mapping logic and generate_translation helper.")

    # 4. Final check of SFT Config (already done but let's ensure evaluation args are there)
    # They should be there from the restore_notebook.py run.

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook audit and fixes completed.")

if __name__ == "__main__":
    audit_and_fix_notebook()
