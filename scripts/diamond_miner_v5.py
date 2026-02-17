import google.generativeai as genai
import pandas as pd
import os
import time
import random
import concurrent.futures
import threading

# CONFIGURACIÓN
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY:
    raise ValueError("❌ FALTA API KEY: Configura la variable de entorno GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

# MODELOS
TEACHER_MODEL_NAME = "models/gemini-3-pro-preview"
STUDENT_MODEL_NAME = "models/gemini-3-flash-preview"

# CONFIGURACIÓN CIENTÍFICA BILINGÜE
LANG_CONFIG = {
    "nah": {
        "output_file": "data/gold/diamond_nah_v1_partial.parquet",
        "domains": [
            "Medicina: Diagnóstico de diabetes y síntomas (Nahuatl)",
            "Legal: Derechos constitucionales en comunidades indígenas",
            "Cultura: Filosofía de la dualidad (Ometeotl)",
            "Dialecto: Variantes de la Huasteca Hidalguense",
            "Gramática: Uso del reverencial (-tzin)",
            "Vida Diaria: Mercado y regateo"
        ],
        "teacher_prompt": """
        Act as a linguistic expert PhD in Nahuatl (Uto-Aztecan).
        Generate 1 highly complex Spanish -> Nahuatl sentence pair.
        Focus: Agglutinative morphology and classical/modern fusion.
        Output format: SPANISH|NAHUATL
        """
    },
    "myn": {
        "output_file": "data/gold/diamond_myn_v1_partial.parquet",
        "domains": [
            "Meliponicultura: Abejas Xunaan Kab y floración",
            "Geología: Cenotes, suelo k'ankab y selva baja",
            "Medicina Maya: J-Men, hierbabuena y ruda",
            "Matemáticas: Sistema vigesimal y cuentas lunares",
            "Gramática: Estructura VOS (Verbo-Objeto-Sujeto)",
            "Vida Diaria: Vaquería y jarana"
        ],
        "teacher_prompt": """
        Act as a linguistic expert PhD in Yucatec Maya (Maaya T'aan).
        Generate 1 highly complex Spanish -> Maya sentence pair.
        Focus: Correct glottal stops ('), tone, and VOS syntax.
        Output format: SPANISH|MAYA
        """
    }
}

stats = {"nah": 0, "myn": 0}
lock = threading.Lock()

def get_teacher_seed(lang, domain):
    """El Maestro (Pro) genera semilla. (LIMITADO A 10s)"""
    # 🛑 ESTRICTO: 10 segundos de espera por hilo.
    time.sleep(10) 
    
    config = LANG_CONFIG[lang]
    model = genai.GenerativeModel(TEACHER_MODEL_NAME)
    prompt = f"""
    {config['teacher_prompt']}
    Topic: {domain}
    Requirement: Pure JSON is not needed here, just the pipe separated format.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        parts = text.split("|")
        if len(parts) >= 2:
            return parts[:2]
    except Exception as e:
        print(f"⚠️ Teacher Error ({lang}): {e}")
    return None

def get_student_variations(lang, es_seed, target_seed):
    """El Estudiante (Flash) genera variantes."""
    model = genai.GenerativeModel(STUDENT_MODEL_NAME)
    lang_name = "Nahuatl" if lang == "nah" else "Maya Yucateco"
    
    prompt = f"""
    Eres un experto nativo de {lang_name}.
    Frase base:
    ES: "{es_seed}"
    TARGET: "{target_seed}"

    Genera 5 variaciones ricas lingüísticamente.
    Cambia sujeto, tiempo o dialecto. Mantén el significado 'Doctorado'.
    
    Formato JSON Array:
    [ {{"es": "...", "target": "..."}}, ... ]
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        import json
        return json.loads(response.text)
    except Exception as e:
        return []

def mine_batch(worker_id):
    lang = "nah" if worker_id % 2 == 0 else "myn"
    config = LANG_CONFIG[lang]
    domain = random.choice(config['domains'])
    
    # 1. Teacher
    seed = get_teacher_seed(lang, domain)
    if not seed: return []
    es_seed, target_seed = seed
    
    # 2. Student
    variations = get_student_variations(lang, es_seed, target_seed)
    
    # 3. Format
    results = []
    # Seed
    results.append({
        "es": es_seed.strip(),
        "target": target_seed.strip(),
        "lang": lang,
        "source": "Gemini-3-Pro",
        "domain": domain,
        "type": "seed_phd"
    })
    # Variations
    for v in variations:
        if "es" in v and "target" in v:
            results.append({
                "es": v["es"],
                "target": v["target"],
                "lang": lang,
                "source": "Gemini-3-Flash",
                "domain": domain,
                "type": "augmentation"
            })
            
    with lock:
        stats[lang] += len(results)
        total = stats["nah"] + stats["myn"]
        if total % 10 == 0:
            print(f"💎 Progress: NAH={stats['nah']} | MAYA={stats['myn']} | Total={total}")
            
    return results

def main():
    print(f"🚀 Iniciando DIAMOND MINE V5 (Rate Limit Protected)...")
    
    all_data_nah = []
    all_data_myn = []
    
    # 🛑 HARD LIMIT: 5 Workers
    # Esto asegura que nunca pasamos de ~20 RPM globalmente.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(mine_batch, i) for i in range(100000)]
        
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            if not results: continue
            
            lang = results[0]['lang']
            if lang == "nah":
                all_data_nah.extend(results)
            else:
                all_data_myn.extend(results)
            
            if (len(all_data_nah) + len(all_data_myn)) % 25 == 0:
                if all_data_nah:
                    pd.DataFrame(all_data_nah).to_parquet(LANG_CONFIG['nah']['output_file'])
                if all_data_myn:
                    pd.DataFrame(all_data_myn).to_parquet(LANG_CONFIG['myn']['output_file'])

if __name__ == "__main__":
    main()
