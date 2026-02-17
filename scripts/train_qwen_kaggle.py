# 📦 INSTALACIÓN (Ejecutar en celda 1)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# ⚙️ CONFIGURACIÓN DE LA MISIÓN
# ==========================================
MAX_SEQ_LENGTH = 2048 # Qwen soporta hasta 128k, pero en Kaggle T4 nos quedamos en 2048/4096 por VRAM
DTYPE = None # Auto detección (Float16 para T4)
LOAD_IN_4BIT = True # Obligatorio para 7B en T4

import glob
import os

# Auto-detect dataset path (Busca cualquier parquet en los inputs)
possible_files = glob.glob("/kaggle/input/**/*.parquet", recursive=True)

if possible_files:
    # Preferimos 'train_v1' si hay varios
    train_files = [f for f in possible_files if "train" in f]
    if train_files:
        DATASET_PATH = train_files[0]
    else:
        DATASET_PATH = possible_files[0]
    print(f"✅ Dataset encontrado en: {DATASET_PATH}")
else:
    print("❌ ALERTA: No se encontró ningún archivo .parquet en /kaggle/input")
    print("Asegurate de haber añadido el dataset en el panel derecho 'Add Input'.")
    DATASET_PATH = "train_v1.parquet" # Fallback local

OUTPUT_DIR = "qwen_nahuatl_finetune"

# ==========================================
# 1. CARGAR MODELO (Qwen 2.5 7B)
# ==========================================
print("⏳ Cargando Qwen 2.5 7B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-Instruct-bnb-4bit", # ¡Qwen 3 Reciente!
    # Nota: Si falla, usaremos "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# Configurar adaptadores LoRA (El cerebro nuevo)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rango de atención. 16 es estándar, 64 es 'God Mode' (más VRAM)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = "unsloth", # Ahorra VRAM brutalmente
    random_state = 3407,
    use_rslora = False,
    loftq_config = None, 
)

# ==========================================
# 2. CARGAR INTELIGENCIA (Dataset)
# ==========================================
print(f"📚 Cargando datos desde {DATASET_PATH}...")
try:
    dataset = load_dataset("parquet", data_files=DATASET_PATH, split="train")
except:
    # Fallback si no está la ruta exacta, intentar local
    dataset = load_dataset("parquet", data_files="train_v1.parquet", split="train")

# Formato de Chat para Qwen (Instruction Tuning)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a professional translator fluent in Spanish, Nahuatl, and Maya. Translate the following text accurately, preserving nuance and dialect.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    inputs       = examples["es"]
    outputs      = examples["nah"] # O 'myn' si entrenas Maya
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# ==========================================
# 3. ENTRENAMIENTO (La Fragua)
# ==========================================
print("🔥 Iniciando Entrenamiento...")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # True acelera pero a veces confunde en datasets pequeños
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # 2x4 = Batch size simulado de 8
        warmup_steps = 100,
        max_steps = 0, 
        num_train_epochs = 3, # Para 24h, 3 eras es buen comienzo
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # Ahorro VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        save_strategy = "steps",
        save_steps = 500, # Guardar cada 500 pasos por si Kaggle muere
        report_to = "none", # O "wandb" si tienes cuenta
    ),
)

trainer_stats = trainer.train()

# ==========================================
# 4. GUARDADO (El Tesoro)
# ==========================================
print("💾 Guardando modelo...")
model.save_pretrained(f"{OUTPUT_DIR}_lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_lora")

# Opcional: Guardar en GGUF para usar en celular directo
# model.save_pretrained_gguf(f"{OUTPUT_DIR}_gguf", tokenizer, quantization_method = "q4_k_m")

print("✅ ¡Misión Cumplida!")
