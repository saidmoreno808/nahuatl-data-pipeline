import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import sacrebleu
from tqdm import tqdm
import os

# --- CONFIGURACIÓN ---
# 1. Nombre del modelo base
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 2. Carpeta donde pusiste los 4 archivos que descargaste
ADAPTER_PATH = "./nahuatl_adapters" 

# 3. Ruta de tu dataset de prueba (CSV o Parquet)
TEST_DATA_PATH = "test_v1.parquet" 

def load_sota_model():
    print("⏳ Cargando modelo base Qwen 2.5 7B (en 4-bit para ahorrar RAM)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("🚀 Inyectando el conocimiento de Náhuatl (Adapters)...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    print("🧠 Cargando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    return model, tokenizer

def generate_translation(model, tokenizer, text):
    prompt = f"Traduce de Español a Náhuatl:\nEspañol: {text}\nNáhuatl:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la respuesta después de "Náhuatl:"
    if "Náhuatl:" in decoded:
        return decoded.split("Náhuatl:")[-1].strip()
    return decoded.strip()

def run_benchmark():
    # Cargar datos
    if TEST_DATA_PATH.endswith('.parquet'):
        df = pd.read_parquet(TEST_DATA_PATH)
    else:
        df = pd.read_csv(TEST_DATA_PATH)
    
    # Asumimos que las columnas son 'es' (fuente) y 'nah' (referencia)
    # O ajusta según tu dataset ('source', 'target', etc.)
    source_col = 'es' if 'es' in df.columns else 'source'
    target_col = 'nah' if 'nah' in df.columns else 'target'
    
    data = df.to_dict('records')[:100] # Prueba con los primeros 100 para rapidez
    
    model, tokenizer = load_sota_model()
    
    refs = []
    hyps = []
    
    print(f"🔍 Evaluando {len(data)} ejemplos...")
    for item in tqdm(data):
        prediction = generate_translation(model, tokenizer, item[source_col])
        refs.append(item[target_col])
        hyps.append(prediction)
    
    # Calcular métricas
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    
    print("\n" + "="*30)
    print("🏆 RESULTADOS FINALES SOTA")
    print("="*30)
    print(f"✨ chrF++: {chrf.score:.2f}")
    print(f"✨ BLEU:   {bleu.score:.2f}")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
