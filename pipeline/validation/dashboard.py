import argparse
from pathlib import Path
from .metrics import MetricsCalculator

def run_dashboard(gold_file: str):
    path = Path(gold_file)
    if not path.exists():
        print(f"Archivo no encontrado: {path}")
        return

    print(f"Generando reporte para {path}...")
    metrics = MetricsCalculator(path)
    report_path = path.parent / f"{path.stem}_report.json"
    report = metrics.generate_report(report_path)
    
    print("\n=== REPORTE DE CALIDAD ===")
    print(f"Total Oraciones: {report['total_sentences']}")
    print(f"TTR Náhuatl: {report['ttr_nah']:.4f}")
    print(f"Longitud Promedio (Náh): {report['length_stats_nah']['mean']:.1f} palabras")
    print(f"Reporte guardado en: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="Ruta al archivo JSONL Gold")
    args = parser.parse_args()
    run_dashboard(args.gold_file)
