import google.generativeai as genai
import pandas as pd
import os
import time
import random
import concurrent.futures
from datetime import datetime

# CONFIGURACIÓN
# ¡OJO! Exporta tu API KEY antes: set GOOGLE_API_KEY=AIzaSy...
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY:
    raise ValueError("❌ FALTA API KEY: Configura la variable de entorno GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

# MODELOS (SOTA Dec 2025)
TEACHER_MODEL_NAME = "models/gemini-3-pro-preview"
STUDENT_MODEL_NAME = "models/gemini-3-flash-preview"

# DOMINIOS CIENTÍFICOS (PhD Curriculum)
DOMAINS = [
    "Medicina: Diagnóstico de diabetes y síntomas",
    "Medicina: Anatomía interna y dolor",
    "Legal: Derechos constitucionales en comunidades indígenas",
    "Legal: Defensa en juicio oral y agrario",
    "Ciencia: Fotosíntesis y biología de plantas locales",
    "Ciencia: Astronomía básica y fases lunares",
    "Cultura: Ritual de Xantolo y significado profundo",
    "Cultura: Filosofía de la dualidad (Ometeotl)",
    "Dialecto: Variantes de la Huasteca Hidalguense",
    "Dialecto: Variantes del Centro (Milpa Alta)",
    "Gramática: Uso del reverencial (-tzin)",
    "Gramática: Estructuras contrafactuales ('hubiera')",
]

# TARGET
TARGET_PAIRS = 5000 # Lote inicial de prueba (luego subimos a 200k)
OUTPUT_FILE = "data/gold/diamond_v1_partial.parquet"

def get_teacher_seed(domain):
    """El Maestro (Pro) genera un ejemplo complejo y perfecto."""
    model = genai.GenerativeModel(TEACHER_MODEL_NAME)
    prompt = f"""
    Act as a linguistic expert PhD in Nahuatl.
    Generate 1 highly complex, grammatically perfect sentence pair (Spanish -> Nahuatl).
    
    Topic: {domain}
    Requirements:
    - Use agglutinative structures correctly.
    - If medical/legal, use precise terminology or loanwords adapted correctly.
    - Output format: SPANISH|NAHUATL
    - No extra text. Just the pair.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "|" in text:
            return text.split("|")
    except Exception as e:
        print(f"⚠️ Teacher Error: {e}")
    return None

def get_student_variations(es_seed, nah_seed):
    """El Estudiante (Flash) genera variantes masivas."""
    model = genai.GenerativeModel(STUDENT_MODEL_NAME)
    prompt = f"""
    Eres un asistente nativo de Nahuatl.
    Tengo esta frase base:
    Español: "{es_seed}"
    Nahuatl: "{nah_seed}"

    Genera 5 variaciones de esta frase cambiando:
    1. El sujeto o el objeto.
    2. El tiempo gramatical.
    3. Un sinónimo dialectal.
    
    Formato de salida (JSON Array):
    [
        {{"es": "...", "nah": "..."}},
        ...
    ]
    Solo JSON puro.
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        import json
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Student Error: {e}")
        return []

def mine_diamond_batch(batch_id):
    """Pipeline de un Worker."""
    domain = random.choice(DOMAINS)
    
    # 1. Teacher Step
    seed = get_teacher_seed(domain)
    if not seed: return []
    es_seed, nah_seed = seed
    
    # 2. Student Step
    variations = get_student_variations(es_seed, nah_seed)
    
    # 3. Format
    results = []
    # Agregamos la semilla original (Calidad Pro)
    results.append({
        "es": es_seed.strip(),
        "nah": nah_seed.strip(),
        "source": "Gemini-3-Pro",
        "domain": domain,
        "type": "seed"
    })
    # Agregamos las variaciones (Volumen Flash)
    for v in variations:
        if "es" in v and "nah" in v:
            results.append({
                "es": v["es"],
                "nah": v["nah"],
                "source": "Gemini-3-Flash",
                "domain": domain,
                "type": "augmentation"
            })
            
    print(f"💎 Batch {batch_id} ({domain}): +{len(results)} pares")
    return results

def main():
    print(f"🚀 Iniciando Diamond Mine v2...")
    print(f"   - Teacher: {TEACHER_MODEL_NAME}")
    print(f"   - Student: {STUDENT_MODEL_NAME}")
    
    all_data = []
    
    # Multithreading para saturar la API (IO Bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(mine_diamond_batch, i) for i in range(100)] # 100 batches iniciales
        
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            all_data.extend(data)
            
            # Guardado progresivo
            if len(all_data) % 50 == 0:
                df = pd.DataFrame(all_data)
                df.to_parquet(OUTPUT_FILE)
                print(f"💾 Checkpoint guardado: {len(df)} filas.")

    # Guardado Final
    df = pd.DataFrame(all_data)
    df.to_parquet(OUTPUT_FILE)
    print(f"✨ MINADO COMPLETADO. Total: {len(df)} pares.")
    print(f"Archivo: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
