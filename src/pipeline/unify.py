"""
Unified Pipeline

Modern refactored pipeline that produces identical results to legacy
scripts/unify_datasets.py while using clean, testable architecture.

This module implements the Bronze -> Silver -> Diamond -> Gold lakehouse
pattern with proper normalization, deduplication, and splitting.

Example:
    >>> pipeline = UnifiedPipeline()
    >>> pipeline.run()
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

from src.models.enums import DataLayer, DataSource
from src.models.schemas import Record
from src.transforms.deduplicators import Deduplicator
from src.transforms.normalizers import TextNormalizer
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedPipeline:
    """
    End-to-end pipeline for dataset unification.

    Loads from Silver + Diamond layers, normalizes, deduplicates,
    splits, and saves to Gold layer.

    Example:
        >>> pipeline = UnifiedPipeline()
        >>> stats = pipeline.run()
        >>> print(f"Processed {stats['total_records']} records")
    """

    def __init__(
        self,
        silver_dir: Path = None,
        diamond_dir: Path = None,
        gold_dir: Path = None,
        seed: int = None,
    ):
        """
        Initialize pipeline.

        Args:
            silver_dir: Path to Silver layer (defaults to data/silver)
            diamond_dir: Path to Diamond layer (defaults to data/diamond)
            gold_dir: Path to Gold layer (defaults to data/gold)
            seed: Random seed for reproducible splits (defaults to 42)
        """
        settings = get_settings()

        self.silver_dir = silver_dir or Path("data/silver")
        self.diamond_dir = diamond_dir or Path("data/diamond")
        self.gold_dir = gold_dir or Path("data/gold")
        self.seed = seed if seed is not None else settings.seed

        # Initialize transforms
        self.normalizer = TextNormalizer()
        self.deduplicator = Deduplicator(case_sensitive=False)

        logger.info(
            f"Pipeline initialized",
            extra={
                "silver_dir": str(self.silver_dir),
                "diamond_dir": str(self.diamond_dir),
                "gold_dir": str(self.gold_dir),
                "seed": self.seed,
            },
        )

    def run(self) -> Dict:
        """
        Run the full pipeline.

        Returns:
            dict: Pipeline statistics
        """
        logger.info("Starting unified pipeline")

        # Step 1: Load records
        records = self._load_all_records()
        logger.info(f"Loaded {len(records)} raw records")

        # Step 2: Normalize
        records = self._normalize_records(records)
        logger.info(f"Normalized {len(records)} records")

        # Step 3: Deduplicate (prioritize Diamond > Silver)
        records = self.deduplicator.deduplicate(records, keep="last")
        logger.info(f"Deduplicated to {len(records)} unique records")

        # Step 4: Split
        train, val, test = self._split_records(records)
        logger.info(
            f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
        )

        # Step 5: Save
        self._save_splits(train, val, test)
        logger.info("Pipeline complete")

        # Return stats
        stats = {
            "total_records": len(records),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "nahuatl_count": sum(1 for r in records if r.has_nahuatl()),
            "maya_count": sum(1 for r in records if r.has_maya()),
        }

        return stats

    def _load_all_records(self) -> List[Record]:
        """Load records from Silver and Diamond layers."""
        records = []

        # Load Silver layer (lower priority)
        silver_records = self._load_layer(
            self.silver_dir,
            layer=DataLayer.SILVER,
        )
        records.extend(silver_records)
        logger.info(f"Loaded {len(silver_records)} Silver records")

        # Load Diamond layer (higher priority)
        diamond_records = self._load_layer(
            self.diamond_dir,
            layer=DataLayer.DIAMOND,
        )
        records.extend(diamond_records)
        logger.info(f"Loaded {len(diamond_records)} Diamond records")

        return records

    def _load_layer(self, directory: Path, layer: DataLayer) -> List[Record]:
        """
        Load all records from a layer directory.

        Args:
            directory: Directory to scan
            layer: Layer to assign to loaded records

        Returns:
            list: Loaded records
        """
        records = []

        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return records

        # Load JSONL files
        for jsonl_file in directory.glob("*.jsonl"):
            layer_records = self._load_jsonl_file(jsonl_file, layer)
            records.extend(layer_records)

        # Load JSON files (legacy dumps)
        for json_file in directory.glob("*.json"):
            layer_records = self._load_json_file(json_file, layer)
            records.extend(layer_records)

        return records

    def _load_jsonl_file(
        self,
        file_path: Path,
        layer: DataLayer,
    ) -> List[Record]:
        """Load records from a JSONL file."""
        records = []
        filename = file_path.name

        logger.debug(f"Loading JSONL: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)

                        # Convert to Record using legacy format handler
                        record = Record.from_legacy_format(data)
                        record.layer = layer
                        record.origin_file = filename

                        # Only keep records with Spanish + (Náhuatl or Maya)
                        if record.has_spanish() and record.has_translation_pair():
                            records.append(record)
                    except json.JSONDecodeError as e:
                        logger.debug(
                            f"JSON decode error in {filename}:{line_num}: {e}"
                        )
                        continue
                    except Exception as e:
                        logger.debug(
                            f"Error processing record in {filename}:{line_num}: {e}"
                        )
                        continue
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

        logger.debug(f"Loaded {len(records)} records from {filename}")
        return records

    def _load_json_file(
        self,
        file_path: Path,
        layer: DataLayer,
    ) -> List[Record]:
        """Load records from a JSON dump file."""
        records = []
        filename = file_path.name

        logger.debug(f"Loading JSON dump: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both list and dict with 'items' key
            items = data.get('items', []) if isinstance(data, dict) else data

            for item in items:
                try:
                    # Handle nested 'original' key (Py-Elotl format)
                    if 'original' in item:
                        orig = item['original']
                        # Handle 'sp' key (alternative for Spanish)
                        record_data = {
                            'es': orig.get('es') or orig.get('sp'),
                            'nah': orig.get('nah'),
                            'myn': orig.get('myn'),
                        }
                    else:
                        record_data = item

                    record = Record.from_legacy_format(record_data)
                    record.layer = layer
                    record.origin_file = filename

                    if record.has_spanish() and record.has_translation_pair():
                        records.append(record)
                except Exception as e:
                    logger.debug(f"Error processing item in {filename}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error reading JSON dump {filename}: {e}")

        logger.debug(f"Loaded {len(records)} records from {filename}")
        return records

    def _normalize_records(self, records: List[Record]) -> List[Record]:
        """Normalize all text fields in records."""
        from src.transforms.normalizers import normalize_record

        normalized = []
        for record in records:
            try:
                normalized_record = normalize_record(record, self.normalizer)
                normalized.append(normalized_record)
            except Exception as e:
                logger.debug(f"Error normalizing record: {e}")
                continue

        return normalized

    def _split_records(
        self,
        records: List[Record],
    ) -> Tuple[List[Record], List[Record], List[Record]]:
        """
        Split records into train/val/test sets.

        Uses same split ratios as legacy (90/5/5) with same seed for
        reproducibility.

        Args:
            records: Records to split

        Returns:
            tuple: (train, val, test) record lists
        """
        settings = get_settings()

        # Set seed for reproducibility (matches legacy)
        random.seed(self.seed)

        # Shuffle indices
        indices = list(range(len(records)))
        random.shuffle(indices)

        # Calculate split points
        total = len(records)
        train_end = int(total * settings.train_ratio)
        val_end = int(total * (settings.train_ratio + settings.val_ratio))

        # Split
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train = [records[i] for i in train_indices]
        val = [records[i] for i in val_indices]
        test = [records[i] for i in test_indices]

        return train, val, test

    def _save_splits(
        self,
        train: List[Record],
        val: List[Record],
        test: List[Record],
    ):
        """Save train/val/test splits to Gold layer."""
        self.gold_dir.mkdir(parents=True, exist_ok=True)

        self._save_jsonl(train, self.gold_dir / "train_v1.jsonl")
        self._save_jsonl(val, self.gold_dir / "validation_v1.jsonl")
        self._save_jsonl(test, self.gold_dir / "test_v1.jsonl")

    def _save_jsonl(self, records: List[Record], path: Path):
        """Save records to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                # Convert to dict and remove None values
                data = record.to_dict()
                clean_data = {k: v for k, v in data.items() if v is not None}
                f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(records)} records to {path}")


# CLI entry point
if __name__ == "__main__":
    from src.utils.logger import configure_root_logger

    configure_root_logger()

    pipeline = UnifiedPipeline()
    stats = pipeline.run()

    print("\n=== Pipeline Complete ===")
    print(f"Total unique records: {stats['total_records']}")
    print(f"  Train:    {stats['train_count']}")
    print(f"  Val:      {stats['val_count']}")
    print(f"  Test:     {stats['test_count']}")
    print(f"\nLanguage distribution:")
    print(f"  Nahuatl:  {stats['nahuatl_count']}")
    print(f"  Maya:     {stats['maya_count']}")
