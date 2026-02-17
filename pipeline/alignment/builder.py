import json
import pandas as pd
from pathlib import Path
import logging

GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

class GoldBuilder:
    def __init__(self):
        self.output_dir = GOLD_DIR

    def save_jsonl(self, data: list, filename: str):
        """Guarda datos en formato JSONL."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"[GOLD] Guardado JSONL: {path}")

    def save_parquet(self, data: list, filename: str):
        """Guarda datos en formato Parquet."""
        path = self.output_dir / filename
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        logger.info(f"[GOLD] Guardado Parquet: {path}")
