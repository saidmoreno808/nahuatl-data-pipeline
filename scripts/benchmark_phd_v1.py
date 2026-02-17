import os
import json
import time
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import sacrebleu
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

# ==========================================
# 🔬 CONFIGURACIÓN CIENTÍFICA (Benchmarking)
# ==========================================
load_dotenv()

# Rutas de Modelos y Datos
QWEN_BASE = "Qwen/Qwen2.5-7B-Instruct"
QWEN_ADAPTERS = "checkpoint-500" 
GOLD_CORPUS = "data/gold/train_sft_v2.parquet"
OUTPUT_RESULTS = "benchmark_phd_results.csv"

# Identificadores de Modelos SOTA
GEMINI_MODEL = "models/gemini-3-pro-preview"
GPT_MODEL = "gpt-4o" 
JUDGE_MODEL = "gemini-1.5-flash" 

# Parámetros de Inferencia
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0 
TOP_P = 0.9

# Configuración de APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Intentar cargar SentenceTransformers para similitud semántica
try:
    from sentence_transformers import SentenceTransformer, util
    ST_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except ImportError:
    print("⚠️ sentence-transformers no instalado. Saltando métrica semántica.")
    ST_MODEL = None

# ==========================================
# ⚖️ LÓGICA DEL JUEZ (LLM-Judge)
# ==========================================

def get_judge_score(source, reference, hyp, model_name):
    """Evalúa una traducción usando la rúbrica de doctorado."""
    model = genai.GenerativeModel(JUDGE_MODEL)
    rubric = """
    Puntuación 1-5:
    5: Nativo/Académico - Perfecto, morfología intacta.
    4: Fluido - Correcto, lenguaje natural.
    3: Comprensible - Errores menores de gramática.
    2: Fragmentado - Calco del español, difícil de entender.
    1: Ininteligible - Alucinación o pérdida total de sentido.
    """
    prompt = f"""
    Actúa como un lingüista experto en Náhuatl. Evalúa la siguiente traducción:
    ORIGINAL (ES): {source}
    REFERENCIA: {reference}
    MODELO ({model_name}): {hyp}

    Rúbrica:
    {rubric}

    Responde SOLO con el número de la puntuación (ej. 5).
    """
    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        # Buscar el primer dígito en el texto
        import re
        match = re.search(r'\d', score_text)
        if match:
            return int(match.group())
    except Exception as e:
        print(f"Error en LLM-Judge: {e}")
    return np.nan

# ==========================================
# 🛠️ UTILIDADES DE INFERENCIA
# ==========================================

def load_qwen_sota():
    """Carga Qwen con adaptadores en 4 bits."""
    print(f"⏳ Cargando Qwen 2.5 7B + Adapters desde {QWEN_ADAPTERS}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, QWEN_ADAPTERS)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
    return model, tokenizer

def predict_qwen(model, tokenizer, text):
    """Inferencia local con Qwen."""
    prompt = f"<start_of_turn>user\nTraduce al Náhuatl Clásico:\n{text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            temperature=0.1
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "model" in decoded:
        return decoded.split("model")[-1].strip()
    return decoded.strip()

def predict_gemini(text):
    """Inferencia vía API Gemini 3 Pro."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"Traduce fielmente del español al náhuatl clásico. Solo la traducción:\n\n'{text}'"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR_GEMINI: {str(e)}"

def predict_openai(text):
    """Inferencia vía API OpenAI (ChatGPT)."""
    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Eres un lingüista experto en náhuatl. Traduce al náhuatl clásico."},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR_OPENAI: {str(e)}"

# ==========================================
# 📊 PIPELINE PRINCIPAL DE BENCHMARK
# ==========================================

def run_benchmark(n_samples=100):
    print(f"🚀 Iniciando Benchmark de Nivel Doctoral ({n_samples} frases)...")
    
    if not os.path.exists(GOLD_CORPUS):
        # Intentar ruta alternativa si no existe
        alt_path = "data/modelo_nahuatl/train_sft_v2.parquet"
        if os.path.exists(alt_path):
            GOLD_CORPUS = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el corpus en {GOLD_CORPUS}")
    
    df_gold = pd.read_parquet(GOLD_CORPUS)
    test_set = df_gold.sample(n_samples, random_state=42)
    
    qwen_model, qwen_tok = load_qwen_sota()
    results = []
    
    for idx, row in tqdm(test_set.iterrows(), total=n_samples, desc="Evaluando"):
        source = row.get('input', row.get('es', ''))
        reference = row.get('output', row.get('nah', ''))
        
        # Inferencia
        p_qwen = predict_qwen(qwen_model, qwen_tok, source)
        p_gemini = predict_gemini(source)
        p_gpt = predict_openai(source)
        
        # Evaluación Cualitativa (Ciega)
        j_qwen = get_judge_score(source, reference, p_qwen, "Qwen-SOTA")
        j_gemini = get_judge_score(source, reference, p_gemini, "Gemini-3-Pro")
        j_gpt = get_judge_score(source, reference, p_gpt, "ChatGPT-4o")
        
        # Similitud Semántica
        s_qwen, s_gemini, s_gpt = np.nan, np.nan, np.nan
        if ST_MODEL:
            ref_emb = ST_MODEL.encode(reference, convert_to_tensor=True)
            s_qwen = util.cos_sim(ref_emb, ST_MODEL.encode(p_qwen, convert_to_tensor=True)).item()
            s_gemini = util.cos_sim(ref_emb, ST_MODEL.encode(p_gemini, convert_to_tensor=True)).item()
            s_gpt = util.cos_sim(ref_emb, ST_MODEL.encode(p_gpt, convert_to_tensor=True)).item()

        results.append({
            "source": source,
            "reference": reference,
            "qwen_pred": p_qwen,
            "gemini_pred": p_gemini,
            "gpt_pred": p_gpt,
            "qwen_score": j_qwen,
            "gemini_score": j_gemini,
            "gpt_score": j_gpt,
            "qwen_sim": s_qwen,
            "gemini_sim": s_gemini,
            "gpt_sim": s_gpt
        })
        time.sleep(0.5)

    df_res = pd.DataFrame(results)
    
    # Métricas Léxicas
    def score_corpus(hyps, refs):
        clean_hyps = [h if "ERROR_" not in h else "" for h in hyps]
        refs_fmt = [[r] for r in refs]
        return sacrebleu.corpus_chrf(clean_hyps, refs_fmt).score

    c_qwen = score_corpus(df_res['qwen_pred'], df_res['reference'])
    c_gemini = score_corpus(df_res['gemini_pred'], df_res['reference'])
    c_gpt = score_corpus(df_res['gpt_pred'], df_res['reference'])
    
    print("\n" + "="*65)
    print("🏆 RESULTADOS FINALES (ESTÁNDAR CIENTÍFICO)")
    print("="*65)
    print(f"MÉTRICA         | QWEN (Tuyo) | GEMINI 3 PRO | ChatGPT")
    print("-" * 65)
    print(f"chrF++ (Léx)   | {c_qwen:11.2f} | {c_gemini:12.2f} | {c_gpt:7.2f}")
    if ST_MODEL:
        print(f"Sim Semántica  | {df_res['qwen_sim'].mean():11.3f} | {df_res['gemini_sim'].mean():12.3f} | {df_res['gpt_sim'].mean():7.3f}")
    print(f"LLM-Judge (1-5)| {df_res['qwen_score'].mean():11.2f} | {df_res['gemini_score'].mean():12.2f} | {df_res['gpt_score'].mean():7.2f}")
    print("="*65)
    
    df_res.to_csv(OUTPUT_RESULTS, index=False)
    print(f"✅ Resultados guardados en {OUTPUT_RESULTS}")

if __name__ == "__main__":
    run_benchmark(100)
