import google.generativeai as genai
import pandas as pd
import time
import random
import os
import json
import sys
import uuid
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN INTELIGENTE (Kaggle vs Colab)
# ==========================================
# Identificador único para este "trabajador"
WORKER_ID = str(uuid.uuid4())[:4]
OUTPUT_FILENAME = f"diamond_layer_maya_{WORKER_ID}.jsonl"
OUTPUT_PATH = OUTPUT_FILENAME # Default local

# Detección de entorno
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("☁️ Detectado Google Colab. Montando Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    # Guardar directo en una carpeta segura
    BASE_DIR = "/content/drive/MyDrive/Maya_AI"
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)
    print(f"✅ Guardando en Drive: {OUTPUT_PATH}")

    # En Colab usas userdata o manual
    from google.colab import userdata
    try:
        GEMINI_KEY = userdata.get('gemini_api_key')
    except:
        GEMINI_KEY = "PON_TU_API_KEY_AQUI_SI_FALLA_USERDATA"
else:
    # Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        GEMINI_KEY = secrets.get_secret("gemini_api_key")
    except:
        # Fallback local
        import os
        GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or "TU_API_KEY_AQUI"

genai.configure(api_key=GEMINI_KEY)

# MODELO
MODEL_NAME = 'gemini-1.5-pro' # O gemini-2.5-flash si está disponible
TARGET_TOTAL = 15000
BATCH_SIZE = 50

# ==========================================
# TEMARIO MAYA YUCATECO (Maayat'aan)
# ==========================================
TEMAS = [
    "Vida Diaria (Familia, Comida, Casa)",
    "Emociones y Sentimientos",
    "Naturaleza (Animales, Plantas, Clima, Selva)",
    "Ciencia y Tecnología Moderna",
    "Noticias Actuales y Política",
    "Historia y Cultura Maya",
    "Filosofía y Cosmovisión",
    "Comercio, Mercado y Dinero",
    "Salud, Cuerpo y Medicina Tradicional",
    "Educación y Escuela",
    "Artes, Música y Danza",
    "Mitos, Leyendas y Cuentos",
    "Campo, Milpa y Agricultura"
]

ESTILOS = [
    "Formal y respetuoso",
    "Casual y coloquial",
    "Poético y metafórico",
    "Directo e informativo"
]

def get_prompt(topic, style, count=10):
    return f"""
    Eres un lingüista experto hablante nativo del Maya Yucateco (Maayat'aan).
    Tu tarea es generar un dataset paralelo de alta calidad para entrenar una IA.

    TEMA: {topic}
    ESTILO: {style}
    CANTIDAD: {count} pares de oraciones.
    REGLAS:
    1. Genera oraciones en ESPAÑOL y su traducción experta al MAYA YUCATECO.
    2. El Maya debe ser gramaticalmente correcto, usando ortografía moderna estandarizada (normas de 1984).
    3. Marca explícitamente las glotalizaciones (k', t', ch', p', tz') y vocales rearticuladas (a'a, e'e, etc.).
    4. NO uses préstamos del español si existe una palabra maya válida (ej: usa 'kíimak óol' para feliz).
    5. Salida estricta en formato JSON (lista de objetos):
    [
        {{"es": "El sol está quemando fuerte", "myn": "Jach chich u k'a'amkach ooxol le k'iino'"}},
        {{"es": "Me duele la cabeza", "myn": "Yaj in pool"}},
        ...
    ]
    SOLO JSON. Nada de markdown.
    """

def generate_batch():
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)
    except:
        print(f"⚠️ Modelo {MODEL_NAME} no respondio, switch a gemini-1.5-flash")
        model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)

    data_buffer = []

    # Revisar si ya existe para retomar
    initial_count = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                initial_count = sum(1 for _ in f)
        except:
            initial_count = 0

    pbar = tqdm(total=TARGET_TOTAL, initial=initial_count, desc=f"💎 Minero Maya {WORKER_ID}")

    while initial_count + len(data_buffer) < TARGET_TOTAL:
        topic = random.choice(TEMAS)
        style = random.choice(ESTILOS)

        try:
            # Invocar
            response = model.generate_content(
                get_prompt(topic, style, count=BATCH_SIZE),
                generation_config={"response_mime_type": "application/json"}
            )

            # Limpieza
            raw_text = response.text.replace("```json", "").replace("```", "").strip()

            try:
                batch_data = json.loads(raw_text)
            except:
                if "[" in raw_text and "]" in raw_text:
                    start = raw_text.find("[")
                    end = raw_text.rfind("]") + 1
                    batch_data = json.loads(raw_text[start:end])
                else:
                    # JSON Roto, saltamos
                    continue

            # Validar
            valid_batch = []
            if isinstance(batch_data, list):
                for item in batch_data:
                    # Adaptamos para aceptar 'myn' o 'nah' por si el modelo alucina keys
                    # pero forzamos estandarizacion
                    es = item.get('es')
                    myn = item.get('myn') or item.get('nah') # Fallback por si acaso

                    if es and myn:
                        valid_batch.append({
                            "es": es.strip(),
                            "myn": myn.strip()
                        })

            if not valid_batch:
                continue

            # GUARDADO SEGURO
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                for item in valid_batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            new_count = len(valid_batch)
            pbar.update(new_count)
            data_buffer.extend(valid_batch)

            time.sleep(2)

        except Exception as e:
            print(f"⚠️ Pausa por error: {e}")
            time.sleep(5)

    pbar.close()
    print(f"✅ ¡Minero Maya {WORKER_ID} terminó! Archivo guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_batch()
