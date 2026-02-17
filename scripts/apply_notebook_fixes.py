import json
import os

notebook_path = "scripts/Kaggle_Nahuatl_Qwen_Training.ipynb"

def update_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 2 (Training Logic) - Smart Loader
    # Target lines around 117-145 in the original file
    smart_loader_code = [
        "# ==========================================\n",
        "# 2. SMART DATASET LOADER + SPLIT\n",
        "# ==========================================\n",
        "print(\"🔍 Buscando datasets en /kaggle/input...\")\n",
        "\n",
        "def find_split_files(base_path=\"/kaggle/input\"):\n",
        "    import glob\n",
        "    possible_files = glob.glob(f\"{base_path}/**/*.*\", recursive=True)\n",
        "    train_file = next((f for f in possible_files if \"train_v1.parquet\" in f), None)\n",
        "    val_file = next((f for f in possible_files if \"validation_v1.parquet\" in f), None)\n",
        "    return train_file, val_file, possible_files\n",
        "\n",
        "train_path, val_path, all_files = find_split_files()\n",
        "\n",
        "if train_path and val_path:\n",
        "    print(f\"✅ Detectados splits dedicados:\\n - Train: {train_path}\\n - Val: {val_path}\")\n",
        "    dataset_train = load_dataset(\"parquet\", data_files=train_path, split=\"train\")\n",
        "    dataset_val = load_dataset(\"parquet\", data_files=val_path, split=\"train\")\n",
        "    train_dataset = dataset_train\n",
        "    eval_dataset = dataset_val\n",
        "else:\n",
        "    print(\"⚠️ Splits no encontrados. Buscando el archivo más grande...\")\n",
        "    data_files = [f for f in all_files if f.endswith(\".parquet\") or f.endswith(\".jsonl\") or f.endswith(\".csv\")]\n",
        "    if not data_files:\n",
        "        print(\"❌ DATASET NO ENCONTRADO - Usando dummy\")\n",
        "        # Create dummy if everything fails\n",
        "        dummy_data = {\"instruction\": [\"test\"]*10, \"input\": [\"test\"]*10, \"output\": [\"test\"]*10}\n",
        "        from datasets import Dataset\n",
        "        dataset = Dataset.from_dict(dummy_data)\n",
        "        dataset = dataset.train_test_split(test_size=0.1)\n",
        "        train_dataset, eval_dataset = dataset[\"train\"], dataset[\"test\"]\n",
        "    else:\n",
        "        data_files.sort(key=lambda x: os.path.getsize(x), reverse=True)\n",
        "        DATASET_PATH = data_files[0]\n",
        "        print(f\"✅ Usando archivo más grande: {DATASET_PATH}\")\n",
        "        ext = DATASET_PATH.split(\".\")[-1]\n",
        "        loader = \"json\" if ext == \"jsonl\" else ext\n",
        "        dataset = load_dataset(loader, data_files=DATASET_PATH, split=\"train\")\n",
        "        dataset = dataset.train_test_split(test_size=0.05, seed=42)\n",
        "        train_dataset, eval_dataset = dataset[\"train\"], dataset[\"test\"]\n",
        "\n",
        "print(f\"📊 Dataset cargado -> Train: {len(train_dataset)} | Val: {len(eval_dataset)}\")\n"
    ]

    # Find the data loading section in Cell 2
    cell2 = nb['cells'][2]
    source = cell2['source']
    
    start_marker = "# ==========================================\n"
    data_marker = "# 2. SMART DATASET LOADER + SPLIT\n"
    
    new_source = []
    skip = False
    replaced = False
    
    for line in source:
        if "# 2. SMART DATASET LOADER" in line or "# 2. CARGA DATASET" in line: # Try to match either
            new_source.extend(smart_loader_code)
            skip = True
            replaced = True
            continue
        if skip:
            if "# ==========================================\n" in line and "3. ENTRENAMIENTO" in line:
                skip = False
                new_source.append("\n") # spacer
                new_source.append(line)
            continue
        new_source.append(line)
    
    if replaced:
        cell2['source'] = new_source
        print("Updated Training Cell (Cell 2).")

    # 2. Add Evaluation Cell (Cell 4) with the fix
    eval_cell_source = [
        "# ==========================================\n",
        "# 📊 EVALUACIÓN Y MÉTRICAS (chrF++ & BLEU)\n",
        "# ==========================================\n",
        "import json\n",
        "import torch\n",
        "import gc\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "from peft import PeftModel\n",
        "from tqdm import tqdm\n",
        "import sacrebleu\n",
        "import os\n",
        "import glob\n",
        "\n",
        "CHECKPOINT_PATH = f\"{OUTPUT_DIR}\" # Usa el directorio de salida del entrenamiento\n",
        "TEST_DATA_PATH = \"/kaggle/input/benchmark-results-csv\" \n",
        "\n",
        "def load_data(path):\n",
        "    # ✅ FIX: Handle directory vs file\n",
        "    if os.path.exists(path) and os.path.isdir(path):\n",
        "        csv_files = glob.glob(os.path.join(path, \"*.csv\"))\n",
        "        if csv_files:\n",
        "            path = csv_files[0]\n",
        "            print(f\"📂 Detectado archivo: {path}\")\n",
        "    \n",
        "    if not os.path.exists(path):\n",
        "        print(f\"⚠️ No se encontró {path}. Usando fallback o dummy...\")\n",
        "        return []\n",
        "    \n",
        "    data = []\n",
        "    if path.endswith(\".csv\"):\n",
        "        import csv\n",
        "        with open(path, 'r', encoding='utf-8') as f:\n",
        "            reader = csv.DictReader(f)\n",
        "            for row in reader:\n",
        "                data.append({\n",
        "                    \"es\": row.get(\"source\", \"\"),\n",
        "                    \"nah\": row.get(\"reference\", \"\"),\n",
        "                    \"gemini_pred\": row.get(\"gemini_pred\", \"\"),\n",
        "                    \"sota_pred\": row.get(\"sota_pred\", \"\")\n",
        "                })\n",
        "    else:\n",
        "        with open(path, 'r', encoding='utf-8') as f:\n",
        "            data = [json.loads(line) for line in f]\n",
        "    return data\n",
        "\n",
        "def main_eval():\n",
        "    data = load_data(TEST_DATA_PATH)\n",
        "    if not data: \n",
        "        print(\"❌ Sin datos para evaluar.\")\n",
        "        return\n",
        "        \n",
        "    # Generar y calcular (Asegurarse de que el modelo esté cargado)\n",
        "    # Si se corre justo después del training, el model ya está en memoria\n",
        "    refs = []\n",
        "    hyps_qwen = []\n",
        "    \n",
        "    print(\"🚀 Generando traducciones...\")\n",
        "    for item in tqdm(data[:50]): # Limitado para velocidad en pruebas\n",
        "        if not item.get('es'): continue\n",
        "        hyp = generate_translation(model, tokenizer, item['es'])\n",
        "        refs.append(item.get('nah', ''))\n",
        "        hyps_qwen.append(hyp)\n",
        "    \n",
        "    chrf_q = sacrebleu.corpus_chrf(hyps_qwen, [refs])\n",
        "    print(f\"\\n🤖 RESULTADOS QWEN 2.5 7B: chrF++: {chrf_q.score:.2f}\")\n",
        "\n",
        "main_eval()\n"
    ]

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": eval_cell_source
    }

    # Add as final cell
    nb['cells'].append(new_cell)
    print("Added Evaluation Cell (Cell 4).")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook()
