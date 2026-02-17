#!/usr/bin/env python3
"""
Script para generar datos DPO usando Gemini Flash en Kaggle
Uso: python kaggle_dpo_script.py --input train_v1.jsonl --output dpo_pairs.jsonl --limit 800
"""

import os
import json
import time
import argparse
import random
from typing import List, Dict
from pathlib import Path

try:
    import google.generativeai as genai
    from tqdm import tqdm
except ImportError:
    print("❌ Instalando dependencias...")
    os.system("pip install -q google-generativeai tqdm")
    import google.generativeai as genai
    from tqdm import tqdm


class GeminiDPOGenerator:
    """Generador optimizado de datos DPO con Gemini"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", rpm_limit: int = 15):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.rpm_limit = rpm_limit
        self.request_times = []
        self.total_requests = 0
        self.errors = 0

    def _wait_for_rate_limit(self):
        """Control inteligente de rate limiting"""
        now = time.time()
        # Limpiar requests antiguos (más de 60 segundos)
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0]) + 1
            if sleep_time > 0:
                print(f"⏸️  Rate limit alcanzado, esperando {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.request_times.append(time.time())
        self.total_requests += 1

    def generate_dpo_pair(self, spanish: str, nahuatl: str) -> Dict:
        """Genera un par DPO (chosen/rejected) para un ejemplo"""

        prompt = f"""Eres un experto traductor español-náhuatl clásico con profundo conocimiento lingüístico.

TAREA: Evalúa y mejora la siguiente traducción.

Español: "{spanish}"
Náhuatl (traducción actual): "{nahuatl}"

INSTRUCCIONES:
1. Analiza la calidad de la traducción actual (gramática, vocabulario, naturalidad)
2. Genera UNA traducción alternativa que sea:
   - Gramaticalmente correcta en náhuatl clásico
   - Pero MENOS natural o precisa que la original
3. Determina cuál es mejor: la original o la alternativa

Responde SOLO con JSON válido:
{{
  "alternative_translation": "traducción alternativa (menor calidad)",
  "original_score": 8,
  "alternative_score": 6,
  "better_option": "original",
  "reasoning": "breve explicación de por qué una es mejor"
}}
"""

        self._wait_for_rate_limit()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=512,
                )
            )

            # Extraer JSON
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)

            # Crear par DPO
            chosen = nahuatl if result.get('better_option') == 'original' else result.get('alternative_translation', nahuatl)
            rejected = result.get('alternative_translation', '') if result.get('better_option') == 'original' else nahuatl

            return {
                'prompt': f"Traduce del español al náhuatl: {spanish}",
                'chosen': chosen,
                'rejected': rejected,
                'metadata': {
                    'original_es': spanish,
                    'original_nah': nahuatl,
                    'evaluation': result,
                    'source': 'gemini-dpo-generation'
                }
            }

        except Exception as e:
            self.errors += 1
            print(f"\n❌ Error en generación: {e}")
            return None

    def generate_synthetic_example(self, context_examples: List[Dict], temperature: float = 0.9) -> Dict:
        """Genera un nuevo ejemplo sintético basado en contexto"""

        samples = random.sample(context_examples, min(3, len(context_examples)))
        context = "\n".join([f"ES: {ex['es']}\nNAH: {ex['nah']}" for ex in samples])

        prompt = f"""Eres un experto en náhuatl clásico.

CONTEXTO - Ejemplos reales de traducción:
{context}

TAREA: Crea UN par de traducción español-náhuatl NUEVO que:
1. Sea temáticamente similar a los ejemplos
2. Use estructuras gramaticales del náhuatl clásico
3. Sea natural y correcto

Responde SOLO con JSON:
{{
  "es": "frase en español",
  "nah": "traducción correcta al náhuatl"
}}
"""

        self._wait_for_rate_limit()

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=256,
                )
            )

            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)

            if result.get('es') and result.get('nah'):
                return {
                    'es': result['es'],
                    'nah': result['nah'],
                    'layer': 'synthetic_gemini',
                    'origin_file': 'gemini_generation',
                    'source': 'gemini-1.5-flash'
                }

        except Exception as e:
            self.errors += 1
            print(f"\n❌ Error en sintético: {e}")
            return None


def load_jsonl(filepath: str) -> List[Dict]:
    """Carga un archivo JSONL"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """Guarda datos en formato JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generador de datos DPO con Gemini')
    parser.add_argument('--input', type=str, required=True, help='Archivo JSONL de entrada')
    parser.add_argument('--output', type=str, default='dpo_pairs.jsonl', help='Archivo de salida DPO')
    parser.add_argument('--limit', type=int, default=800, help='Máximo de ejemplos a procesar')
    parser.add_argument('--synthetic', type=int, default=150, help='Ejemplos sintéticos a generar')
    parser.add_argument('--api-key', type=str, help='API Key de Gemini (o usar variable GEMINI_API_KEY)')
    parser.add_argument('--rpm', type=int, default=15, help='Requests por minuto (límite)')

    args = parser.parse_args()

    # Obtener API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')

    # Intentar obtener de Kaggle Secrets
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret("GEMINI_API_KEY")
            print("✅ API Key cargada desde Kaggle Secrets")
        except:
            pass

    if not api_key:
        print("❌ Error: No se encontró GEMINI_API_KEY")
        print("   Opciones:")
        print("   1. Agregar como Kaggle Secret: GEMINI_API_KEY")
        print("   2. Variable de entorno: export GEMINI_API_KEY=...")
        print("   3. Parámetro: --api-key YOUR_KEY")
        return

    print("🚀 Iniciando generación DPO con Gemini Flash")
    print(f"📂 Input: {args.input}")
    print(f"📂 Output: {args.output}")
    print(f"🎯 Límite: {args.limit} ejemplos DPO + {args.synthetic} sintéticos")
    print(f"⏱️  Rate limit: {args.rpm} rpm")
    print()

    # Cargar dataset
    print("📥 Cargando dataset...")
    data = load_jsonl(args.input)
    print(f"✅ {len(data):,} ejemplos cargados")

    # Inicializar generador
    generator = GeminiDPOGenerator(api_key, rpm_limit=args.rpm)

    # Seleccionar muestra (priorizar "diamond")
    diamond = [ex for ex in data if ex.get('layer') == 'diamond']
    others = [ex for ex in data if ex.get('layer') != 'diamond']

    sample = diamond[:int(args.limit * 0.7)] + others[:int(args.limit * 0.3)]
    random.shuffle(sample)
    sample = sample[:args.limit]

    print(f"\n📊 Muestra seleccionada:")
    print(f"   Diamond: {sum(1 for ex in sample if ex.get('layer') == 'diamond')}")
    print(f"   Otros: {sum(1 for ex in sample if ex.get('layer') != 'diamond')}")

    # Generar pares DPO
    print(f"\n🔄 Generando {len(sample)} pares DPO...")
    print(f"⏱️  Tiempo estimado: {len(sample) / args.rpm:.1f} minutos\n")

    dpo_pairs = []
    checkpoint_interval = 100

    for idx, example in enumerate(tqdm(sample, desc="DPO generation")):
        pair = generator.generate_dpo_pair(example['es'], example['nah'])

        if pair:
            dpo_pairs.append(pair)

        # Checkpoint cada 100
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_file = f"{args.output}.checkpoint_{idx+1}"
            save_jsonl(dpo_pairs, checkpoint_file)
            print(f"\n💾 Checkpoint guardado: {checkpoint_file}")

    print(f"\n✅ Pares DPO generados: {len(dpo_pairs)}")
    print(f"❌ Errores: {generator.errors}")
    print(f"📊 Tasa de éxito: {len(dpo_pairs)/(len(dpo_pairs)+generator.errors)*100:.1f}%")

    # Guardar DPO pairs
    save_jsonl(dpo_pairs, args.output)
    print(f"💾 Guardado: {args.output}")

    # Generar sintéticos si hay requests disponibles
    remaining = max(0, 950 - generator.total_requests)
    synthetic_count = min(remaining, args.synthetic)

    if synthetic_count > 0:
        print(f"\n🎨 Generando {synthetic_count} ejemplos sintéticos...")

        synthetic_data = []
        for i in tqdm(range(synthetic_count), desc="Sintéticos"):
            example = generator.generate_synthetic_example(data)
            if example:
                synthetic_data.append(example)

        # Guardar sintéticos
        synthetic_file = args.output.replace('.jsonl', '_synthetic.jsonl')
        save_jsonl(synthetic_data, synthetic_file)
        print(f"✅ Sintéticos generados: {len(synthetic_data)}")
        print(f"💾 Guardado: {synthetic_file}")

        # Crear dataset extendido
        extended_data = data + synthetic_data
        extended_file = args.output.replace('.jsonl', '_extended.jsonl')
        save_jsonl(extended_data, extended_file)
        print(f"💾 Dataset extendido: {extended_file} ({len(extended_data)} ejemplos)")

    # Estadísticas finales
    stats = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'gemini-1.5-flash',
        'input_file': args.input,
        'original_size': len(data),
        'dpo_pairs': len(dpo_pairs),
        'synthetic_examples': synthetic_count if synthetic_count > 0 else 0,
        'total_api_calls': generator.total_requests,
        'errors': generator.errors,
        'success_rate': f"{len(dpo_pairs)/(len(dpo_pairs)+generator.errors)*100:.2f}%",
        'cost': '$0.00 (free tier)',
        'execution_time_minutes': generator.total_requests / args.rpm
    }

    stats_file = args.output.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("📊 RESUMEN FINAL")
    print('='*60)
    for key, value in stats.items():
        print(f"{key:.<35} {value}")
    print('='*60)
    print(f"\n✅ Proceso completado exitosamente")
    print(f"📦 Archivos generados:")
    print(f"   - {args.output}")
    if synthetic_count > 0:
        print(f"   - {synthetic_file}")
        print(f"   - {extended_file}")
    print(f"   - {stats_file}")


if __name__ == '__main__':
    main()
