
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import sacrebleu
import os

# ==========================================
# 🧹 LIMPIEZA DE MEMORIA (Crucial si corres en el mismo notebook)
# ==========================================
print("🧹 Limpiando memoria GPU previa...")
try:
    del model
    del trainer
except:
    pass
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Apunta directamente a la carpeta que se ve en tu imagen
CHECKPOINT_PATH = "qwen25_7b_nahuatl_v3/checkpoint-200" 
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
# TEST_DATA_PATH = "data/gold/benchmark_test_set_es_nah.jsonl"

# 🔥 FIX: Ajustamos la ruta para que coincida con tu dataset subido
# En Kaggle los datasets subidos aparecen en /kaggle/input/NOMBRE_DEL_DATASET/archivo
TEST_DATA_PATH = "/kaggle/input/benchmark-results-csv/benchmark_results.csv"

def load_data(path):
    if not os.path.exists(path):
        print(f"⚠️ No se encontró {path}. Creando dataset dummy para prueba...")
        return [
            {"es": "Hola mundo", "nah": "Piali taltikpak"},
            {"es": "El gato come", "nah": "In miston tlakua"}
        ]
    
    data = []
    if path.endswith(".csv"):
        import csv
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Mapear columnas del CSV a formato estándar
                item = {
                    "es": row.get("source", ""),
                    "nah": row.get("reference", ""),
                    "gemini_pred": row.get("gemini_pred", ""),
                    "sota_pred": row.get("sota_pred", "")
                }
                data.append(item)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    return data

def load_model(checkpoint_path):
    print(f"⏳ Cargando Base: {BASE_MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"⏳ Cargando Adaptador: {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_translation(model, tokenizer, text):
    alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate Spanish to Native Language

### Input:
{text}

### Response:
"""
    inputs = tokenizer(alpaca_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            use_cache=True,
            temperature=0.3, # Bajo para evaluación determinista
            top_p=0.9,
            do_sample=True 
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la respuesta (despues de ### Response:)
    if "### Response:" in full_text:
        return full_text.split("### Response:")[-1].strip()
    return full_text.strip()

def calculate_metrics(refs, hyps):
    # chrF
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return chrf, bleu

def main():
    # 1. Cargar Datos
    data = load_data(TEST_DATA_PATH)
    print(f"📊 Evaluando sobre {len(data)} ejemplos.")
    
    # 2. Cargar Modelo
    model, tokenizer = load_model(CHECKPOINT_PATH)
    
    # 3. Generar
    refs = []
    hyps_qwen = []
    
    print("🚀 Generando traducciones...")
    for item in tqdm(data):
        if 'es' not in item or 'nah' not in item: continue
        
        src = item['es']
        ref = item['nah']
        
        hyp = generate_translation(model, tokenizer, src)
        
        refs.append(ref)
        hyps_qwen.append(hyp)
        
    # 4. Calcular Métricas Qwen
    chrf_q, bleu_q = calculate_metrics(refs, hyps_qwen)
    
    print("\n" + "="*40)
    print(f"🤖 RESULTADOS QWEN 2.5 7B (Nahuatl)")
    print("="*40)
    print(f"✅ chrF++: {chrf_q.score:.2f}")
    print(f"✅ BLEU:   {bleu_q.score:.2f}")
    
    # 5. Comparar con Gemini / SOTA (Si existen en el CSV)
    hyps_gemini = [x.get("gemini_pred", "") for x in data]
    hyps_gemini = [h for h in hyps_gemini if h] # Filter empty if any
    
    if len(hyps_gemini) == len(refs):
        chrf_g, bleu_g = calculate_metrics(refs, hyps_gemini)
        print("\n" + "="*40)
        print(f"💎 RESULTADOS GEMINI (Baseline)")
        print("="*40)
        print(f"✅ chrF++: {chrf_g.score:.2f}")
        print(f"✅ BLEU:   {bleu_g.score:.2f}")
        print(f"\n🏆 DIFERENCIA (Qwen - Gemini): {chrf_q.score - chrf_g.score:+.2f} chrF")

    hyps_sota = [x.get("sota_pred", "") for x in data]
    hyps_sota = [h for h in hyps_sota if h]
    
    if len(hyps_sota) == len(refs):
        chrf_s, bleu_s = calculate_metrics(refs, hyps_sota)
        print("\n" + "="*40)
        print(f"⚔️ RESULTADOS SOTA/GEMMA (Anterior)")
        print("="*40)
        print(f"✅ chrF++: {chrf_s.score:.2f}")
        print(f"✅ BLEU:   {bleu_s.score:.2f}")
        print(f"\n🏆 DIFERENCIA (Qwen - SOTA):   {chrf_q.score - chrf_s.score:+.2f} chrF")

if __name__ == "__main__":
    main()
