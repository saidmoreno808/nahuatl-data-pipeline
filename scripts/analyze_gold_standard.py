
import pandas as pd
import numpy as np
import os

FILE_PATH = "data/gold/train_phd_v1.parquet"

def analyze_phd_quality():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: No se encontró {FILE_PATH}")
        return


    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"🔬 Iniciando Auditoría Científica de: {FILE_PATH}")
    log("="*60)

    try:
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        log(f"❌ Error crítico al leer parquet: {e}")
        return

    # 1. Volumetría
    total_rows = len(df)
    log(f"📊 Total de Muestras: {total_rows:,}")
    
    # 2. Schema Check
    cols = df.columns.tolist()
    log(f"🗂️ Columnas detectadas: {cols}")
    
    expected_cols = ['es', 'nah']
    has_input_output = 'input' in cols and 'output' in cols
    has_es_nah = 'es' in cols and 'nah' in cols
    
    if not (has_input_output or has_es_nah):
        log("⚠️ ALERTA: El esquema no es estándar (es/nah o input/output). Esto complicará el entrenamiento.")
    else:
        log("✅ Esquema compatible con SFT.")

    # 3. Limpieza (Nulls & Empties)
    # Estandarizar
    if has_es_nah:
        src_col, tgt_col = 'es', 'nah'
    elif has_input_output:
        src_col, tgt_col = 'input', 'output'
    else:
        src_col, tgt_col = cols[0], cols[1]

    null_count = df.isnull().sum().sum()
    empty_src = df[df[src_col].astype(str).str.strip() == ""].shape[0]
    empty_tgt = df[df[tgt_col].astype(str).str.strip() == ""].shape[0]
    
    log(f"\n🗑️ Calidad de Datos:")
    log(f"   - Valores Nulos: {null_count}")
    log(f"   - Source Vacíos: {empty_src}")
    log(f"   - Target Vacíos: {empty_tgt}")
    
    # 4. Duplicados
    exact_dupes = df.duplicated().sum()
    src_dupes = df.duplicated(subset=[src_col]).sum()
    log(f"   - Duplicados Exactos: {exact_dupes} ({exact_dupes/total_rows:.1%})")
    log(f"   - Inputs Repetidos (Ambigüedad): {src_dupes} ({src_dupes/total_rows:.1%})")

    # 5. Análisis de Longitud (Tokens aproximados)
    df['src_len'] = df[src_col].astype(str).apply(len)
    df['tgt_len'] = df[tgt_col].astype(str).apply(len)
    
    log(f"\n📏 Distribución de Longitud (Caracteres):")
    log(f"   - Source: Avg={df['src_len'].mean():.1f} | Max={df['src_len'].max()} | Min={df['src_len'].min()}")
    log(f"   - Target: Avg={df['tgt_len'].mean():.1f} | Max={df['tgt_len'].max()} | Min={df['tgt_len'].min()}")
    
    short_samples = df[df['tgt_len'] < 10].shape[0]
    log(f"   - Muestras 'basura' (<10 chars): {short_samples:,}")

    # 6. Muestreo Visual
    log("\n👀 Muestreo Aleatorio (Verificación Humana):")
    sample = df.sample(5)
    for idx, row in sample.iterrows():
        log(f"   Input:  {row[src_col][:80]}...")
        log(f"   Output: {row[tgt_col][:80]}...")
        log("-" * 30)

    # 7. Veredicto del "Doctor"
    log("\n🎓 VEREDICTO DE CALIDAD CIENTÍFICA:")
    score = 100
    if total_rows < 10000: score -= 30
    if null_count > 0: score -= 10
    if exact_dupes > 500: score -= 10
    if src_dupes > 2000: score -= 10
    if short_samples > 1000: score -= 10
    
    log(f"   PUNTUACIÓN: {score}/100")
    if score >= 80:
        log("   ✅ APTO PARA NIVEL DOCTORADO (SOTA)")
    elif score >= 60:
        log("   ⚠️ ACEPTABLE CON RESERVAS (Requiere limpieza)")
    else:
        log("   ❌ NO APTO (Datos sucios o insuficientes)")

    with open("gold_standard_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    analyze_phd_quality()
