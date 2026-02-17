
import pandas as pd
import numpy as np
import os
import re

INPUT_PATH = "data/gold/train_v1.parquet"
OUTPUT_PATH = "data/gold/train_phd_v1.parquet"

def clean_text(text):
    if text is None: return ""
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

def polish_dataset():
    print(f"💎 Iniciando Pulido de Dataset (PhD Level)...")
    df = pd.read_parquet(INPUT_PATH)
    original_len = len(df)
    
    # 1. Definir columnas (es/nah o input/output)
    cols = df.columns.tolist()
    if 'es' in cols:
        src, tgt = 'es', 'nah'
    else:
        src, tgt = 'input', 'output'

    # 2. Pipeline de Limpieza
    
    # A) Filtrado de Nulos y Vacíos
    df = df.dropna(subset=[src, tgt])
    df[src] = df[src].apply(clean_text)
    df[tgt] = df[tgt].apply(clean_text)
    df = df[ (df[src] != "") & (df[tgt] != "") ]
    step1_len = len(df)
    print(f"   - Nulos/Vacíos eliminados: {original_len - step1_len}")

    # B) Filtrado de "Basura" Corta (< 15 caracteres)
    # Excepción: Si es una palabra clave muy específica, pero para SFT general mejor evitar.
    # Seamos estrictos para SOTA.
    df['src_len'] = df[src].apply(len)
    df['tgt_len'] = df[tgt].apply(len)
    
    df = df[ (df['src_len'] >= 10) & (df['tgt_len'] >= 10) ]
    step2_len = len(df)
    print(f"   - Muestras cortas (<10 chars) eliminadas: {step1_len - step2_len}")

    # C) Deduplicación Inteligente (Resolver Ambigüedad)
    # Si hay inputs repetidos, nos quedamos con el target más largo (asumiendo que es más explicativo)
    # o el que tenga mayor vocabulario único.
    
    # Ordenamos por longitud del Target Descendente
    df = df.sort_values(by='tgt_len', ascending=False)
    
    # Eliminamos duplicados en SOURCE, quedándonos con el primero (el más largo)
    df = df.drop_duplicates(subset=[src], keep='first')
    
    final_len = len(df)
    print(f"   - Duplicados de Input resueltos: {step2_len - final_len}")
    
    # D) Filtrado de caracteres extraños (Opcional pero recomendado para Nahuatl)
    # (Aquí asumimos que el dataset base ya pasó por limpieza básica en Silver)

    # 3. Guardado
    print("="*40)
    print(f"✨ RESULTADO FINAL:")
    print(f"   - Original: {original_len:,}")
    print(f"   - PHD Clean: {final_len:,}")
    print(f"   - Reducción: {100*(1 - final_len/original_len):.1f}% (Grasa eliminada)")
    
    df.drop(columns=['src_len', 'tgt_len'], inplace=True, errors='ignore')
    
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"💾 Guardado en: {OUTPUT_PATH}")
    print("✅ LISTO PARA ENTRENAMIENTO EN A100.")

if __name__ == "__main__":
    polish_dataset()
