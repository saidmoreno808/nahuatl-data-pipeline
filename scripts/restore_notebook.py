import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def restore_and_fix_notebook():
    if not os.path.exists(notebook_path):
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # We need to find the cell that contains the data loader and re-add the training logic
    # Based on the current state, Cell 2 (index 2) has the loader but stopped there.
    cell2 = nb['cells'][2]
    source = cell2['source']

    # Training logic with OOM fixes
    training_logic = [
        "\n",
        "# ==========================================\n",
        "# 3. ENTRENAMIENTO SOTA\n",
        "# ==========================================\n",
        "print(\"🔥 Configurando SOTA Trainer... + W&B\")\n",
        "\n",
        "AUTO_EPOCHS = 1 if len(train_dataset) > 10000 else 3\n",
        "\n",
        "# OPTIMIZACIÓN DE MEMORIA PARA EVITAR OOM\n",
        "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
        "\n",
        "sft_config_args = {\n",
        "    \"output_dir\": OUTPUT_DIR,\n",
        "    \"per_device_train_batch_size\": 1, \n",
        "    \"per_device_eval_batch_size\": 1,   # 🔥 FIX OOM\n",
        "    \"gradient_accumulation_steps\": 8, \n",
        "    \"eval_accumulation_steps\": 4,      # 🔥 FIX OOM\n",
        "    \n",
        "    \"learning_rate\": 1e-4, \n",
        "    \"fp16\": True,                      # 🔥 FIX OOM (T4 GPUs)\n",
        "    \"bf16\": False,\n",
        "    \"max_steps\": -1,\n",
        "    \"num_train_epochs\": AUTO_EPOCHS,\n",
        "    \"logging_steps\": 10,\n",
        "    \"optim\": \"paged_adamw_8bit\",\n",
        "    \n",
        "    \"eval_strategy\": \"steps\",\n",
        "    \"eval_steps\": 100,\n",
        "    \"save_strategy\": \"steps\",\n",
        "    \"save_steps\": 100,\n",
        "    \"load_best_model_at_end\": True, \n",
        "    \"metric_for_best_model\": \"eval_loss\",\n",
        "    \"greater_is_better\": False,\n",
        "    \"save_total_limit\": 2,\n",
        "\n",
        "    \"gradient_checkpointing\": True,\n",
        "    \"neftune_noise_alpha\": 5,\n",
        "    \"report_to\": \"wandb\",     \n",
        "    \"run_name\": \"qwen25-7b-nahuatl-sotav4\",\n",
        "    \"packing\": True\n",
        "}\n",
        "\n",
        "import inspect\n",
        "sft_sig = inspect.signature(SFTConfig)\n",
        "if \"max_seq_length\" in sft_sig.parameters:\n",
        "    sft_config_args[\"max_seq_length\"] = 2048\n",
        "    sft_config_args[\"dataset_text_field\"] = \"text\"\n",
        "\n",
        "training_args = SFTConfig(**sft_config_args)\n",
        "\n",
        "trainer_kwargs = {\n",
        "    \"model\": model,\n",
        "    \"train_dataset\": train_dataset,\n",
        "    \"eval_dataset\": eval_dataset,\n",
        "    \"args\": training_args,\n",
        "}\n",
        "\n",
        "trainer_sig = inspect.signature(SFTTrainer)\n",
        "if \"max_seq_length\" in trainer_sig.parameters:\n",
        "    trainer_kwargs[\"max_seq_length\"] = 2048\n",
        "    trainer_kwargs[\"dataset_text_field\"] = \"text\"\n",
        "\n",
        "if \"processing_class\" in trainer_sig.parameters:\n",
        "    trainer_kwargs[\"processing_class\"] = tokenizer\n",
        "else:\n",
        "    trainer_kwargs[\"tokenizer\"] = tokenizer\n",
        "\n",
        "trainer = SFTTrainer(**trainer_kwargs)\n",
        "\n",
        "print(f\"🚀 Entrenando ({AUTO_EPOCHS} Epochs) con Best Model Loading y W&B... \")\n",
        "trainer.train()\n"
    ]

    # Check if "# 3. ENTRENAMIENTO SOTA" is already there (to avoid duplicates, though unlikely now)
    if not any("# 3. ENTRENAMIENTO SOTA" in line for line in source):
        # Append training logic to Cell 2
        source.extend(training_logic)
        print("Training logic restored and OOM fixes applied to Cell 2.")
    else:
        # If it's there but needs fixes, we'd have to replace. 
        # But given the previous view_file, it's missing.
        print("Training logic already present, check if it needs manual update.")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")

if __name__ == "__main__":
    restore_and_fix_notebook()
