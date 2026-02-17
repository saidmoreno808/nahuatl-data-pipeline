# ============================================================
# 🎓 QWEN 3 32B - BALANCED TRAINING V4 (NAHUATL)
# ============================================================
# Configuración: Transformers + PEFT + RSLoRA + Dual T4 + W&B
# Dataset: 70k Balanceado (59k Gold + 11k DPO sobremuestreado 10x)
# Kaggle GPU: T4 x2 (30GB VRAM Total)
# Modelo: 32B Parameters (High Reasoning Tier)
import os
import subprocess
import sys

# 🚀 ACTUALIZACIÓN CRÍTICA: Forzar versiones SOTA para evitar conflictos de API
print("📦 Actualizando librerías críticas...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "trl", "transformers", "accelerate", "peft", "bitsandbytes"])

# 🔥 FIX NUCLEAR PROTOBUF: Debe estar al inicio absoluto
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 💾 FIX DISCO: Redirigir caché a /kaggle/working
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.makedirs('/kaggle/working/hf_cache', exist_ok=True)

# 🧠 FIX VRAM: Prevenir fragmentación agresiva
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import wandb
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from kaggle_secrets import UserSecretsClient

# 1. Autenticación y Rutas
user_secrets = UserSecretsClient()

# Login Hugging Face
hf_token = user_secrets.get_secret("hf_token")
login(token=hf_token)

# Login W&B
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("wandb_api_key")
wandb.init(project="qwen3-nahuatl-balanced-v4", name="run-balanced-70k-opcionD")

# 🔍 DETECCIÓN DE MODELO (Kaggle Models vs Hugging Face)
MODEL_ID = "Qwen/Qwen3-32B"
LOCAL_PATH = "/kaggle/input/qwen-3/transformers/32b/1"
if os.path.exists(LOCAL_PATH):
    print(f"✅ Detectado modelo local en: {LOCAL_PATH}")
    MODEL_ID = LOCAL_PATH
else:
    print(f"⚠️ No se detectó modelo en /kaggle/input. Se intentará descargar de HF (¡Cuidado con el espacio!)")

# 🎯 DATASET BALANCEADO V4 (70k: 59k Gold + 11k DPO 10x)
DATA_PATH = "/kaggle/input/nahuatl-balanced-corpus-70k/train_balanced_v2.parquet"

# 🔄 CHECKPOINT ANTERIOR (Opcional - para continuar desde checkpoint 800)
RESUME_FROM_CHECKPOINT = None
CHECKPOINT_PATH = "/kaggle/input/nawatl-qwen3-checkpoint-800/checkpoint-800"
if os.path.exists(CHECKPOINT_PATH):
    print(f"✅ Detectado checkpoint anterior: {CHECKPOINT_PATH}")
    print("⚠️  Para continuar desde este checkpoint, descomenta la línea siguiente:")
    # RESUME_FROM_CHECKPOINT = CHECKPOINT_PATH
else:
    print("ℹ️  No se detectó checkpoint anterior. Entrenando desde modelo base.")

OUTPUT_DIR = "./qwen3-32b-nawatl-balanced-v4"

# 2. Configuración Cuantización 4-bit (Crítico para 32B en T4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Turing prefiere fp16 sobre bf16
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,  # 🔥 CRÍTICO: Permite offload a CPU si es necesario
)

# 🧹 LIMPIEZA PRE-CARGA (Crítico para Dual T4)
import gc
gc.collect()
torch.cuda.empty_cache()

# ⚖️ Mapa de Memoria Explícito (Para 32B en 2x T4 de 15GB)
# Ajustado para ser más conservador y permitir offload sin errores
max_memory = {
    0: "13GiB",  # GPU 0
    1: "13GiB",  # GPU 1  
    "cpu": "30GiB"  # 🔥 CRÍTICO: Permitir offload a CPU si es necesario
}

# 3. Cargar Tokenizer y Modelo
print("🚀 Cargando Gigante Qwen 3 32B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",  # Deja que Accelerate optimice la distribución
    max_memory=max_memory,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 🔥 Usar torch_dtype en lugar de dtype
    low_cpu_mem_usage=True,
    offload_folder="offload",  # 🔥 Carpeta para offload temporal
)
model.config.torch_dtype = torch.float16 # Asegurar que la config reporte float16
model = prepare_model_for_kbit_training(model)

# 🔥 FIX NUCLEAR: Eliminar CUALQUIER rastro de BFloat16 en capas no cuantizadas
print("🧹 Limpiando tipos de datos (BFloat16 -> Float16) en capas compatibles (Deep Sea Purge)...")

# 1. Forzar conversión de módulos específicos propensos a quedarse en BF16
for name, module in model.named_modules():
    if any(kw in name.lower() for kw in ["norm", "emb", "ln", "head"]):
        module.to(torch.float16)

# 2. Conversión recursiva de parámetros y buffers restantes
for name, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

for name, buf in model.named_buffers():
    if buf.dtype == torch.bfloat16:
        buf.data = buf.data.to(torch.float16)

# 3. Asegurar que la configuración del modelo esté alineada
model.config.torch_dtype = torch.float16
print("✅ Purga completada. Tipos de datos alineados con FP16.")

# 4. Configuración RSLoRA (No envolver manualmente para evitar merge_and_unload)
peft_config = LoraConfig(
    r=16, # Reducción extrema para optimizar VRAM de estados AdamW
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

# 5. Pipeline de Datos
def formatting_prompts_func(example):
    return f"<|im_start|>user\nTraduce al Náhuatl Clásico:\n{example['es']}<|im_end|>\n<|im_start|>assistant\n{example['nah']}<|im_end|>"

print(f"📊 Cargando Dataset Balanceado V4 desde: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"   ✅ {len(df):,} ejemplos cargados")
print(f"   📈 Composición aproximada: 84% Gold + 16% DPO (10x oversampled)")

dataset = Dataset.from_pandas(df)

# 6. Argumentos de Entrenamiento - OPCIÓN D OPTIMIZADA 🎯
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    
    # 🔥 CAMBIOS OPCIÓN D
    learning_rate=2e-5,  # ⬇️ Reducido de 3e-5 (más conservador)
    num_train_epochs=2,   # ⬇️ Reducido de 3 (balance tiempo/calidad)
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # 📊 LOGGING Y CHECKPOINTS
    logging_steps=10,  # ⬆️ Más frecuente para monitoreo
    save_strategy="steps",
    save_steps=50,  # 🚨 CRÍTICO: Cada 50 pasos para sesiones 12h (antes: 200)
    save_total_limit=3,  # ⬇️ Solo últimos 3 checkpoints (antes: 5)
    
    # 🔥 FIX SINERGIA T4: Desactivamos AMP (fp16/bf16) del Trainer para evitar el GradScaler.
    # El modelo sigue usando float16 para cómputo gracias a BitsAndBytesConfig(bnb_4bit_compute_dtype).
    fp16=False, 
    bf16=False,
    
    optim="paged_adamw_8bit",
    report_to="wandb",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, # Estabilidad adicional
    neftune_noise_alpha=5,
    
    # 🔄 RESUME FROM CHECKPOINT (si está configurado)
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
)

# Logs de configuración
print("\n" + "="*60)
print("⚙️  CONFIGURACIÓN DE ENTRENAMIENTO V4")
print("="*60)
print(f"📂 Dataset: {len(df):,} ejemplos")
print(f"🎓 Épocas: {training_args.num_train_epochs}")
print(f"📈 Learning Rate: {training_args.learning_rate}")
print(f"💾 Checkpoints cada: {training_args.save_steps} steps")
print(f"🔄 Resume desde: {RESUME_FROM_CHECKPOINT if RESUME_FROM_CHECKPOINT else 'Modelo base'}")
print(f"⏱️  Steps por época: ~{len(df) // training_args.gradient_accumulation_steps:,}")
print(f"⏱️  Steps totales: ~{(len(df) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs:,}")
print("="*60 + "\n")

# 7. Inicializar Trainer con Inspección Total (Defensa de Grado Doctoral)
print("🛠️ Configurando SFTTrainer con inspección dinámica...")
import inspect
from trl import SFTTrainer

# Parámetros potenciales
potential_kwargs = {
    "model": model,
    "train_dataset": dataset,
    "peft_config": peft_config,
    "formatting_func": formatting_prompts_func,
    "args": training_args,
    "max_seq_length": 256, # Límite de seguridad para 32B en T4
    "tokenizer": tokenizer,
    "processing_class": tokenizer, # Nuevo nombre en TRL 0.12+
}

# Obtener firma del SFTTrainer
sig = inspect.signature(SFTTrainer.__init__)
valid_params = sig.parameters.keys()

# Construir kwargs finales solo con lo que el Trainer acepte
trainer_kwargs = {k: v for k, v in potential_kwargs.items() if k in valid_params}

# Logs de depuración para el usuario
print(f"✅ Parámetros aceptados por tu versión de TRL: {list(trainer_kwargs.keys())}")

# 🧹 LIMPIEZA PRE-ENTRENAMIENTO (Crítico para 32B)
import gc
gc.collect()
torch.cuda.empty_cache()

# Inicialización final
trainer = SFTTrainer(**trainer_kwargs)

# 8. ¡INICIAR ENTRENAMIENTO!
print("\n" + "🔥"*30)
print("🔥 INICIANDO ENTRENAMIENTO V4 - OPCIÓN D BALANCEADA")
print("🔥"*30 + "\n")
print("📊 Monitoreo en tiempo real:")
print(f"   W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print(f"   Checkpoints: {OUTPUT_DIR}/checkpoint-XXXX")
print("\n⏰ Recuerda: Kaggle tiene límite de 12h. Descarga checkpoints periódicamente.\n")

trainer.train()

# 9. Finalizar
print("\n" + "="*60)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*60)
print("💾 Guardando adaptadores finales...")

trainer.model.save_pretrained(f"{OUTPUT_DIR}_final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")

print(f"📂 Modelo guardado en: {OUTPUT_DIR}_final")
print(f"📊 Logs W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print("\n🎯 Próximos pasos:")
print("   1. Descargar checkpoint final como ZIP")
print("   2. Ejecutar benchmark V8 para validar CHRF >50")
print("   3. Si necesitas continuar, sube checkpoint como dataset")
print("="*60)

wandb.finish()
