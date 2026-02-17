import pandas as pd
import json
from pathlib import Path

class MetricsCalculator:
    def __init__(self, gold_file: Path):
        self.df = pd.read_json(gold_file, lines=True)

    def calculate_ttr(self, text_col="nah"):
        """Calcula Type-Token Ratio."""
        if text_col not in self.df.columns:
            return 0.0
        
        all_text = " ".join(self.df[text_col].astype(str))
        tokens = all_text.split()
        if not tokens:
            return 0.0
        
        types = set(tokens)
        return len(types) / len(tokens)

    def length_distribution(self, text_col="nah"):
        """Calcula estadísticas de longitud."""
        if text_col not in self.df.columns:
            return {}
        
        lengths = self.df[text_col].astype(str).apply(lambda x: len(x.split()))
        return {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median())
        }

    def generate_report(self, output_path: Path):
        """Genera reporte completo."""
        report = {
            "total_sentences": len(self.df),
            "ttr_nah": self.calculate_ttr("nah"),
            "length_stats_nah": self.length_distribution("nah")
        }
        
        if "es" in self.df.columns:
            report["ttr_es"] = self.calculate_ttr("es")
            report["length_stats_es"] = self.length_distribution("es")
            
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report
