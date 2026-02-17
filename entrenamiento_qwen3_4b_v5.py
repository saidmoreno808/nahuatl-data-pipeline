# ============================================================
# 🚀 QWEN3-4B-INSTRUCT-2507 - NAHUATL EXPERT TRAINING V5
# ============================================================
# Configuración: Transformers + PEFT + RSLoRA + Dual T4 + W&B
# Dataset: 70k Balanceado (59k Gold + 11k DPO sobremuestreado 10x)
# Kaggle GPU: T4 x2 (30GB VRAM Total)
# Modelo: 4B Parameters — SIN cuantización (bf16 nativo)
# ============================================================

import os
import subprocess
import sys
import gc
import shutil
import json
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 📦 SECCIÓN 0: INSTALACIÓN DE DEPENDENCIAS
# ─────────────────────────────────────────────────────────────

print("📦 Actualizando librerías críticas...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "trl", "transformers>=4.51.0", "accelerate", "peft",
    "bitsandbytes", "wandb", "huggingface_hub"
])

# ─────────────────────────────────────────────────────────────
# 🔧 SECCIÓN 1: CONFIGURACIÓN DEL ENTORNO
# ─────────────────────────────────────────────────────────────

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.makedirs('/kaggle/working/hf_cache', exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import wandb
from datasets import Dataset
from huggingface_hub import login, HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from kaggle_secrets import UserSecretsClient

# ─────────────────────────────────────────────────────────────
# 🔑 SECCIÓN 2: AUTENTICACIÓN
# ─────────────────────────────────────────────────────────────

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("hf_token")
login(token=hf_token)

# Obtener username de HF para repo de checkpoints
from huggingface_hub import HfApi
api = HfApi()
user_info = api.whoami(token=hf_token)
username = user_info["name"]

os.environ["WANDB_API_KEY"] = user_secrets.get_secret("wandb_api_key")
wandb.init(
    project="qwen3-4b-nahuatl-v5",
    name=f"run-4b-instruct-{datetime.now().strftime('%m%d-%H%M')}",
    tags=["qwen3-4b", "nahuatl", "sft", "v5"]
)

# ─────────────────────────────────────────────────────────────
# 📂 SECCIÓN 3: DETECCIÓN DE MODELO Y DATASET
# ─────────────────────────────────────────────────────────────

# 🔍 Rutas de modelo (Kaggle Models vs HuggingFace)
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_PATHS = [
    "/kaggle/input/qwen3-4b-instruct-2507/transformers/default/1",
    "/kaggle/input/qwen3-4b-instruct-2507",
    "/kaggle/input/wowfattie-qwen3-4b-instruct-2507",
]

model_path = MODEL_ID  # Default: descargar de HF
for path in LOCAL_PATHS:
    if os.path.exists(path):
        # Verificar que tiene archivos de modelo
        files = os.listdir(path) if os.path.isdir(path) else []
        if any(f.endswith(('.safetensors', '.bin', 'config.json')) for f in files):
            model_path = path
            print(f"✅ Modelo local detectado: {path}")
            break
        # Buscar subdirectorios
        for sub in os.listdir(path):
            subpath = os.path.join(path, sub)
            if os.path.isdir(subpath):
                subfiles = os.listdir(subpath)
                if any(f.endswith(('.safetensors', '.bin', 'config.json')) for f in subfiles):
                    model_path = subpath
                    print(f"✅ Modelo local detectado: {subpath}")
                    break
        if model_path != MODEL_ID:
            break

if model_path == MODEL_ID:
    print(f"⚠️ Modelo local no detectado. Descargando de HF: {MODEL_ID}")

# 🎯 Dataset
DATA_PATH = "/kaggle/input/nahuatl-balanced-corpus-70k/train_balanced_v2.parquet"
if not os.path.exists(DATA_PATH):
    # Intentar paths alternativos
    alt_paths = [
        "/kaggle/input/nahuatl-balanced-corpus/train_balanced_v2.parquet",
        "/kaggle/input/train-balanced-v2/train_balanced_v2.parquet",
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            DATA_PATH = alt
            break
    else:
        raise FileNotFoundError(
            f"❌ Dataset no encontrado. Intenté:\n"
            f"  - {DATA_PATH}\n"
            f"  - {chr(10).join(alt_paths)}\n"
            f"Agrega 'nahuatl-balanced-corpus-70k' como dataset en el notebook."
        )

print(f"📂 Modelo: {model_path}")
print(f"📂 Dataset: {DATA_PATH}")

# 🔄 CHECKPOINT ANTERIOR (para resumir entrenamiento)
RESUME_FROM_CHECKPOINT = None

# ──────── Detección de checkpoints locales ────────
CHECKPOINT_PATHS = [
    "/kaggle/input/qwen3-4b-nawatl-checkpoint/checkpoint-latest",
    "/kaggle/input/qwen3-4b-checkpoint",
]
for ckpt in CHECKPOINT_PATHS:
    if os.path.exists(ckpt):
        # Buscar el checkpoint más reciente
        if os.path.isdir(ckpt):
            subdirs = [d for d in os.listdir(ckpt) if d.startswith("checkpoint-")]
            if subdirs:
                latest = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
                RESUME_FROM_CHECKPOINT = os.path.join(ckpt, latest)
            elif os.path.exists(os.path.join(ckpt, "adapter_config.json")):
                RESUME_FROM_CHECKPOINT = ckpt
        if RESUME_FROM_CHECKPOINT:
            print(f"🔄 Checkpoint local detectado: {RESUME_FROM_CHECKPOINT}")
            break

# ──────── Si no hay checkpoint local, buscar en HF Hub ────────
if not RESUME_FROM_CHECKPOINT:
    print("🆕 No hay checkpoint local, verificando HuggingFace Hub...")
    
    try:
        from huggingface_hub import snapshot_download, list_repo_files
        
        # Usar mismo repo que el callback
        HF_REPO_ID = f"{username}/qwen3-4b-nawatl-v5-checkpoints"
        
        # Listar archivos para encontrar checkpoints
        try:
            all_files = list_repo_files(HF_REPO_ID, repo_type="model", token=hf_token)
            
            # Encontrar checkpoints (e.g., checkpoint-100/, checkpoint-200/)
            checkpoints_in_hub = set()
            for file_path in all_files:
                if file_path.startswith("checkpoint-"):
                    ckpt_name = file_path.split("/")[0]
                    checkpoints_in_hub.add(ckpt_name)
            
            if checkpoints_in_hub:
                # Ordenar por número de step
                sorted_ckpts = sorted(
                    checkpoints_in_hub,
                    key=lambda x: int(x.split("-")[1])
                )
                latest_hub_ckpt = sorted_ckpts[-1]
                
                print(f"☁️ Checkpoint encontrado en HF Hub: {latest_hub_ckpt}")
                print(f"   Descargando desde {HF_REPO_ID}...")
                
                # Descargar solo el último checkpoint
                local_dir = f"/kaggle/working/hf_checkpoint"
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    allow_patterns=[f"{latest_hub_ckpt}/**"],
                    local_dir=local_dir,
                    token=hf_token,
                )
                
                RESUME_FROM_CHECKPOINT = os.path.join(local_dir, latest_hub_ckpt)
                print(f"   ✅ Checkpoint descargado: {RESUME_FROM_CHECKPOINT}")
            else:
                print(f"   ℹ️ No hay checkpoints en {HF_REPO_ID}")
                print(f"   🆕 Entrenamiento desde cero")
        
        except Exception as e:
            print(f"   ⚠️ Error consultando HF Hub: {e}")
            print(f"   🆕 Entrenamiento desde cero")
    
    except ImportError:
        print("   ⚠️ huggingface_hub no disponible para auto-descarga")
        print(f"   🆕 Entrenamiento desde cero")

if not RESUME_FROM_CHECKPOINT:
    print("🆕 Entrenamiento desde cero (sin checkpoint previo)")


OUTPUT_DIR = "./qwen3-4b-nawatl-v5"

# ─────────────────────────────────────────────────────────────
# 🤖 SECCIÓN 4: CARGAR MODELO (SIN CUANTIZACIÓN)
# ─────────────────────────────────────────────────────────────

gc.collect()
torch.cuda.empty_cache()

print("🚀 Cargando Qwen3-4B-Instruct-2507 en bf16 NATIVO...")
print(f"   💡 Sin cuantización — 4B params caben en T4x2 (~8GB)")

# Detectar si T4 soporta bf16 (Turing: parcial via Tensor Cores)
# Usamos fp16 que es nativamente soportado y más rápido en T4
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    print(f"   🎮 GPU: {gpu_name} (compute {capability[0]}.{capability[1]})")
    n_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus))
    print(f"   💾 GPUs: {n_gpus}x | VRAM Total: {total_vram / 1e9:.1f} GB")

    # T4 = compute 7.5 → usar fp16 (Tensor Cores optimizados para fp16)
    # A100+ = compute 8.0+ → usar bf16
    USE_BF16 = capability[0] >= 8
    USE_FP16 = not USE_BF16
    COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
    print(f"   ⚡ Precisión: {'bfloat16' if USE_BF16 else 'float16'}")
else:
    raise RuntimeError("❌ No se detectó GPU. Activa T4x2 en Settings.")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # Compatibilidad universal
)

# Verificar carga exitosa
total_params = sum(p.numel() for p in model.parameters())
trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ Modelo cargado: {total_params / 1e9:.2f}B parámetros")
print(f"   📊 Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────
# 🎛️ SECCIÓN 5: CONFIGURACIÓN LoRA (RSLoRA)
# ─────────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=32,                        # ⬆️ Más rank (4B permite más)
    lora_alpha=64,               # Mantener alpha = 2×r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,             # RSLoRA para estabilidad
)

print(f"🎛️ LoRA Config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# ─────────────────────────────────────────────────────────────
# 📊 SECCIÓN 6: PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────

def formatting_prompts_func(example):
    """Formato ChatML compatible con Qwen3-4B-Instruct-2507."""
    return (
        f"<|im_start|>user\n"
        f"Traduce al Náhuatl Clásico:\n{example['es']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{example['nah']}<|im_end|>"
    )

print(f"📊 Cargando Dataset desde: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"   ✅ {len(df):,} ejemplos cargados")

# Verificar columnas
assert 'es' in df.columns, f"❌ Columna 'es' no encontrada. Columnas: {list(df.columns)}"
assert 'nah' in df.columns, f"❌ Columna 'nah' no encontrada. Columnas: {list(df.columns)}"

# Estadísticas rápidas
print(f"   📏 Longitud promedio ES: {df['es'].str.len().mean():.0f} chars")
print(f"   📏 Longitud promedio NAH: {df['nah'].str.len().mean():.0f} chars")

dataset = Dataset.from_pandas(df)

# ─────────────────────────────────────────────────────────────
# ⚙️ SECCIÓN 7: ARGUMENTOS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

# Cálculos de steps
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
EFFECTIVE_BATCH = PER_DEVICE_BATCH * GRAD_ACCUM * max(1, torch.cuda.device_count())
NUM_EPOCHS = 3
STEPS_PER_EPOCH = len(df) // EFFECTIVE_BATCH
TOTAL_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS

print(f"\n{'='*60}")
print(f"⚙️  CONFIGURACIÓN DE ENTRENAMIENTO V5")
print(f"{'='*60}")
print(f"📂 Dataset: {len(df):,} ejemplos")
print(f"🎓 Épocas: {NUM_EPOCHS}")
print(f"📦 Batch: {PER_DEVICE_BATCH}/device × {GRAD_ACCUM} accum × {torch.cuda.device_count()} GPU = {EFFECTIVE_BATCH} efectivo")
print(f"⏱️  Steps por época: ~{STEPS_PER_EPOCH:,}")
print(f"⏱️  Steps totales: ~{TOTAL_STEPS:,}")
print(f"💾 Checkpoints cada: 100 steps (guardando mejores 5)")
print(f"🔄 Resume: {RESUME_FROM_CHECKPOINT or 'Desde cero'}")
print(f"{'='*60}\n")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # 📦 Batch y Épocas
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,

    # 📈 Learning Rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,

    # 💾 Checkpoints — cada 100 steps, guardar mejores 5
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,

    # 📊 Logging
    logging_steps=10,
    logging_first_step=True,
    report_to="wandb",

    # ⚡ Precisión — FP16 para T4 Tensor Cores
    fp16=USE_FP16,
    bf16=USE_BF16,

    # 🧠 Optimización de memoria
    optim="adamw_torch_fused",    # Más rápido que paged_adamw_8bit para 4B
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # 🎯 Regularización
    neftune_noise_alpha=5,
    max_grad_norm=1.0,

    # 🔄 Resume
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT,

    # 📁 Misc
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=42,
)

# ─────────────────────────────────────────────────────────────
# 🏋️ SECCIÓN 8: INICIALIZAR TRAINER
# ─────────────────────────────────────────────────────────────

print("🛠️ Configurando SFTTrainer...")

import inspect

potential_kwargs = {
    "model": model,
    "train_dataset": dataset,
    "peft_config": peft_config,
    "formatting_func": formatting_prompts_func,
    "args": training_args,
    "max_seq_length": 512,  # ⬆️ 2x vs V4 (el modelo soporta 262K)
    "tokenizer": tokenizer,
    "processing_class": tokenizer,
    "dataset_text_field": None,
}

# Compatibilidad dinámica con diferentes versiones de TRL
sig = inspect.signature(SFTTrainer.__init__)
valid_params = sig.parameters.keys()
trainer_kwargs = {k: v for k, v in potential_kwargs.items() if k in valid_params and v is not None}

print(f"✅ Parámetros aceptados: {list(trainer_kwargs.keys())}")

# Limpieza pre-entrenamiento
gc.collect()
torch.cuda.empty_cache()

trainer = SFTTrainer(**trainer_kwargs)

# Verificar parámetros entrenables
trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in trainer.model.parameters())
print(f"🎛️ Parámetros entrenables: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")
print(f"💾 Memoria GPU post-setup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ─────────────────────────────────────────────────────────────
# 🔄 SECCIÓN 9: CALLBACK PARA GUARDAR CHECKPOINTS EN HF HUB
# ─────────────────────────────────────────────────────────────
# Kaggle NO puede montar Google Drive directamente.
# En su lugar, subimos checkpoints a HuggingFace Hub automáticamente
# para que sobrevivan entre sesiones.

from transformers import TrainerCallback

HF_REPO_ID = None  # Se configura abajo

class HubCheckpointCallback(TrainerCallback):
    """Sube cada checkpoint guardado a HuggingFace Hub para persistencia."""

    def __init__(self, repo_id, tokenizer, every_n_saves=1):
        self.repo_id = repo_id
        self.tokenizer = tokenizer
        self.api = HfApi()
        self.every_n_saves = every_n_saves
        self.save_count = 0

    def on_save(self, args, state, control, **kwargs):
        self.save_count += 1
        if self.save_count % self.every_n_saves != 0:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return

        try:
            print(f"\n☁️  Subiendo checkpoint-{state.global_step} a HF Hub ({self.repo_id})...")

            # 🔧 FIX: Corregir adapter_config.json antes de subir
            adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
            CANONICAL_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
            
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    config = json.load(f)
                
                # Reemplazar path local de Kaggle con modelo canónico
                old_base = config.get("base_model_name_or_path", "")
                if "/kaggle/" in old_base or not old_base.startswith("Qwen/"):
                    config["base_model_name_or_path"] = CANONICAL_MODEL_ID
                    
                    with open(adapter_config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    
                    print(f"   🔧 Corregido adapter_config.json: {old_base} → {CANONICAL_MODEL_ID}")
            
            # 🔧 FIX: Corregir README.md también
            readme_path = os.path.join(checkpoint_dir, "README.md")
            if os.path.exists(readme_path):
                import re
                
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                
                # Patrón: buscar línea "base_model: /kaggle/..." en frontmatter YAML
                pattern = r"(base_model:\s*)([^\n]+)"
                
                def replace_base_model(match):
                    prefix = match.group(1)
                    old_value = match.group(2).strip()
                    if "/kaggle/" in old_value or not old_value.startswith("Qwen/"):
                        return f"{prefix}{CANONICAL_MODEL_ID}"
                    return match.group(0)
                
                readme_fixed = re.sub(pattern, replace_base_model, readme_content)
                
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_fixed)
                
                print(f"   🔧 Corregido README.md metadata")


            # Subir archivos del checkpoint
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.repo_id,
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_type="model",
                commit_message=f"Checkpoint step {state.global_step} | loss={state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}",
            )

            # También guardar un archivo de estado para saber cuál es el último
            state_info = {
                "last_checkpoint_step": state.global_step,
                "epoch": state.epoch,
                "total_steps": state.max_steps,
                "timestamp": datetime.now().isoformat(),
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
            }
            state_path = os.path.join(args.output_dir, "hub_state.json")
            with open(state_path, "w") as f:
                json.dump(state_info, f, indent=2)

            self.api.upload_file(
                path_or_fileobj=state_path,
                path_in_repo="hub_state.json",
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Update state: step {state.global_step}",
            )

            print(f"   ✅ Checkpoint-{state.global_step} subido exitosamente")
        except Exception as e:
            print(f"   ⚠️ Error subiendo a Hub (no fatal): {e}")
            print(f"   💡 El checkpoint local sigue guardado en: {checkpoint_dir}")

# Configurar persistencia en HF Hub
try:
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    HF_REPO_ID = f"{username}/qwen3-4b-nawatl-v5-checkpoints"

    # Crear repo si no existe
    try:
        api.create_repo(HF_REPO_ID, repo_type="model", private=True, exist_ok=True)
        print(f"☁️  Persistencia HF Hub activada: {HF_REPO_ID}")
        print(f"   📎 Ver en: https://huggingface.co/{HF_REPO_ID}")

        # Agregar callback al trainer
        hub_callback = HubCheckpointCallback(
            repo_id=HF_REPO_ID,
            tokenizer=tokenizer,
            every_n_saves=1,  # Subir CADA checkpoint (cada 100 steps)
        )
        trainer.add_callback(hub_callback)
    except Exception as e:
        print(f"⚠️ No se pudo crear repo HF: {e}")
        print("   💡 Los checkpoints se guardarán solo localmente.")
except Exception as e:
    print(f"⚠️ HF Hub no disponible: {e}")
    print("   💡 Los checkpoints se guardarán solo localmente.")

# ─────────────────────────────────────────────────────────────
# 🔥 SECCIÓN 10: ¡ENTRENAR!
# ─────────────────────────────────────────────────────────────

print("\n" + "🔥" * 30)
print("🔥 INICIANDO ENTRENAMIENTO V5 - QWEN3-4B NÁHUATL EXPERT")
print("🔥" * 30 + "\n")
print("📊 Monitoreo en tiempo real:")
print(f"   W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print(f"   HF:  https://huggingface.co/{HF_REPO_ID or 'N/A'}")
print(f"   💾 Checkpoints locales: {OUTPUT_DIR}/checkpoint-XXXX")
print(f"\n⏰ Tiempo estimado: ~{TOTAL_STEPS * 10 / 3600:.1f}h ({TOTAL_STEPS} steps × ~10s/step)")
print(f"   Con 12h Kaggle: ~{12 * 3600 / 10:.0f} steps por sesión\n")

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

# ─────────────────────────────────────────────────────────────
# 💾 SECCIÓN 11: GUARDAR MODELO FINAL
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 60)

FINAL_DIR = f"{OUTPUT_DIR}_final"
print(f"💾 Guardando adaptadores finales en: {FINAL_DIR}")
trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# Subir modelo final a HF Hub
if HF_REPO_ID:
    try:
        print(f"☁️ Subiendo modelo FINAL a HF Hub...")
        api.upload_folder(
            folder_path=FINAL_DIR,
            repo_id=HF_REPO_ID,
            path_in_repo="final",
            repo_type="model",
            commit_message="🎉 Modelo final entrenado",
        )
        print(f"   ✅ Modelo final subido a: https://huggingface.co/{HF_REPO_ID}/tree/main/final")
    except Exception as e:
        print(f"   ⚠️ Error subiendo final: {e}")

# Empaquetar como ZIP para descarga directa
print("📦 Empaquetando modelo final como ZIP...")
shutil.make_archive(f"nawatl_qwen3_4b_final", "zip", FINAL_DIR)
print(f"   📥 Descarga: /kaggle/working/nawatl_qwen3_4b_final.zip")

# ─────────────────────────────────────────────────────────────
# 🧪 SECCIÓN 12: BENCHMARK RÁPIDO POST-ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("🧪 BENCHMARK RÁPIDO POST-ENTRENAMIENTO")
print("=" * 60)

test_cases = [
    "El sol calienta la tierra",
    "Los niños juegan en el río",
    "La lluvia cae suavemente sobre las montañas",
    "Mi corazón está lleno de alegría",
    "La flor más bella crece en el jardín",
]

model.eval()
with torch.no_grad():
    for text in test_cases:
        prompt = (
            f"<|im_start|>user\n"
            f"Traduce al Náhuatl Clásico:\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n  🇪🇸 {text}")
        print(f"  🇲🇽 {response.strip()}")

# ─────────────────────────────────────────────────────────────
# 📊 SECCIÓN 13: RESUMEN FINAL
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("📊 RESUMEN FINAL")
print("=" * 60)
print(f"🤖 Modelo: Qwen3-4B-Instruct-2507")
print(f"📂 Dataset: {len(df):,} ejemplos")
print(f"🎓 Épocas completadas: {NUM_EPOCHS}")
print(f"🎛️ LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}")
print(f"💾 Modelo local: {FINAL_DIR}")
print(f"☁️ Modelo HF Hub: https://huggingface.co/{HF_REPO_ID or 'N/A'}")
print(f"📊 W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print(f"\n🎯 Próximos pasos:")
print(f"   1. Descargar nawatl_qwen3_4b_final.zip")
print(f"   2. Ejecutar benchmark completo CHRF/BLEU")
print(f"   3. Si CHRF > 40: Iniciar Fase 2 (DPO)")
print(f"   4. Si necesitas continuar: sube checkpoint como dataset")
print("=" * 60)

wandb.finish()
print("\n✨ ¡Entrenamiento V5 completado exitosamente!")

