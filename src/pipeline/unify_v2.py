"""
Unified Pipeline v2 - Production Ready

Enhanced pipeline with error handling, progress tracking, and metadata persistence.

Features:
- Graceful error handling with custom exceptions
- Progress bars for user feedback (tqdm)
- Metadata tracking (pipeline runs, lineage)
- Batch processing for memory efficiency
- Detailed logging with context

Example:
    >>> pipeline = UnifiedPipeline(show_progress=True)
    >>> stats = pipeline.run()
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

from src.exceptions import DataLoadError, DataTransformError, PipelineError
from src.models.enums import DataLayer, DataSource
from src.models.schemas import Record
from src.transforms.deduplicators import Deduplicator
from src.transforms.normalizers import TextNormalizer
from src.utils.config import get_settings
from src.utils.db import get_db_connection, execute_update, execute_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedPipeline:
    """
    Production-ready unified pipeline with error handling and progress tracking.

    Example:
        >>> pipeline = UnifiedPipeline(show_progress=True, batch_size=1000)
        >>> stats = pipeline.run()
        >>> print(f"Pipeline run ID: {stats['run_id']}")
    """

    def __init__(
        self,
        silver_dir: Path = None,
        diamond_dir: Path = None,
        gold_dir: Path = None,
        seed: int = None,
        show_progress: bool = True,
        batch_size: int = None,
        save_metadata: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            silver_dir: Path to Silver layer
            diamond_dir: Path to Diamond layer
            gold_dir: Path to Gold layer
            seed: Random seed for reproducible splits
            show_progress: Show progress bars (disable for CI/CD)
            batch_size: Batch size for processing (None = process all at once)
            save_metadata: Save pipeline run metadata to database
        """
        settings = get_settings()

        self.silver_dir = silver_dir or Path("data/silver")
        self.diamond_dir = diamond_dir or Path("data/diamond")
        self.gold_dir = gold_dir or Path("data/gold")
        self.seed = seed if seed is not None else settings.seed
        self.show_progress = show_progress
        self.batch_size = batch_size or settings.batch_size
        self.save_metadata = save_metadata

        # Initialize transforms
        self.normalizer = TextNormalizer()
        self.deduplicator = Deduplicator(case_sensitive=False)

        # Pipeline state
        self.run_id: Optional[int] = None
        self.start_time: Optional[float] = None
        self.errors: List[Dict] = []

        logger.info(
            "Pipeline initialized",
            extra={
                "silver_dir": str(self.silver_dir),
                "diamond_dir": str(self.diamond_dir),
                "gold_dir": str(self.gold_dir),
                "seed": self.seed,
                "show_progress": self.show_progress,
                "batch_size": self.batch_size,
            },
        )

    def run(self) -> Dict:
        """
        Run the full pipeline with error handling and metadata tracking.

        Returns:
            dict: Pipeline statistics including run_id, timings, counts

        Raises:
            PipelineError: If pipeline fails critically
        """
        self.start_time = time.time()

        try:
            logger.info("Starting unified pipeline")

            # Create pipeline run record
            if self.save_metadata:
                self.run_id = self._create_pipeline_run()

            # Step 1: Load records
            records = self._load_all_records()
            logger.info(f"Loaded {len(records)} raw records")

            if len(records) == 0:
                raise PipelineError("No records loaded", silver_dir=str(self.silver_dir))

            # Step 2: Normalize
            records = self._normalize_records(records)
            logger.info(f"Normalized {len(records)} records")

            # Step 3: Deduplicate
            initial_count = len(records)
            records = self.deduplicator.deduplicate(records, keep="last")
            duplicates_removed = initial_count - len(records)
            logger.info(
                f"Deduplicated to {len(records)} unique records",
                extra={"duplicates_removed": duplicates_removed},
            )

            # Step 4: Split
            train, val, test = self._split_records(records)
            logger.info(
                f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
            )

            # Step 5: Save
            self._save_splits(train, val, test)
            logger.info("Saved splits to Gold layer")

            # Calculate stats
            duration = time.time() - self.start_time
            stats = {
                "run_id": self.run_id,
                "total_records": len(records),
                "duplicates_removed": duplicates_removed,
                "train_count": len(train),
                "val_count": len(val),
                "test_count": len(test),
                "nahuatl_count": sum(1 for r in records if r.has_nahuatl()),
                "maya_count": sum(1 for r in records if r.has_maya()),
                "duration_seconds": round(duration, 2),
                "errors_count": len(self.errors),
            }

            # Update pipeline run record
            if self.save_metadata:
                self._complete_pipeline_run(stats)

            logger.info(
                "Pipeline complete",
                extra={
                    "duration_seconds": stats["duration_seconds"],
                    "total_records": stats["total_records"],
                },
            )

            return stats

        except Exception as e:
            duration = time.time() - self.start_time if self.start_time else 0

            # Mark pipeline run as failed
            if self.save_metadata and self.run_id:
                self._fail_pipeline_run(str(e), duration)

            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise PipelineError(f"Pipeline execution failed: {e}") from e

    def _create_pipeline_run(self) -> int:
        """Create a pipeline run record in the database."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO pipeline_runs (
                        pipeline_name,
                        status,
                        started_at,
                        config
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        "unified_pipeline",
                        "running",
                        datetime.utcnow().isoformat(),
                        json.dumps({
                            "seed": self.seed,
                            "batch_size": self.batch_size,
                            "silver_dir": str(self.silver_dir),
                            "diamond_dir": str(self.diamond_dir),
                            "gold_dir": str(self.gold_dir),
                        }),
                    ),
                )
                run_id = cursor.lastrowid
                logger.debug(f"Created pipeline run: {run_id}")
                return run_id
        except Exception as e:
            logger.warning(f"Failed to create pipeline run record: {e}")
            return None

    def _complete_pipeline_run(self, stats: Dict):
        """Mark pipeline run as completed with statistics."""
        if not self.run_id:
            return

        try:
            with get_db_connection() as conn:
                execute_update(
                    conn,
                    """
                    UPDATE pipeline_runs
                    SET status = ?,
                        completed_at = ?,
                        records_processed = ?,
                        records_output = ?,
                        duration_seconds = ?
                    WHERE run_id = ?
                    """,
                    (
                        "completed",
                        datetime.utcnow().isoformat(),
                        stats["total_records"],
                        stats["train_count"] + stats["val_count"] + stats["test_count"],
                        stats["duration_seconds"],
                        self.run_id,
                    ),
                )
                logger.debug(f"Updated pipeline run: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to update pipeline run record: {e}")

    def _fail_pipeline_run(self, error_message: str, duration: float):
        """Mark pipeline run as failed."""
        if not self.run_id:
            return

        try:
            with get_db_connection() as conn:
                execute_update(
                    conn,
                    """
                    UPDATE pipeline_runs
                    SET status = ?,
                        completed_at = ?,
                        duration_seconds = ?,
                        error_message = ?
                    WHERE run_id = ?
                    """,
                    (
                        "failed",
                        datetime.utcnow().isoformat(),
                        round(duration, 2),
                        error_message,
                        self.run_id,
                    ),
                )
                logger.debug(f"Marked pipeline run as failed: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to mark pipeline run as failed: {e}")

    def _load_all_records(self) -> List[Record]:
        """Load records from Silver and Diamond layers with progress tracking."""
        records = []

        # Load Silver layer
        if self.show_progress:
            print("\n[1/5] Loading Silver layer...")

        silver_records = self._load_layer(
            self.silver_dir,
            layer=DataLayer.SILVER,
        )
        records.extend(silver_records)
        logger.info(f"Loaded {len(silver_records)} Silver records")

        # Load Diamond layer
        if self.show_progress:
            print("[2/5] Loading Diamond layer...")

        diamond_records = self._load_layer(
            self.diamond_dir,
            layer=DataLayer.DIAMOND,
        )
        records.extend(diamond_records)
        logger.info(f"Loaded {len(diamond_records)} Diamond records")

        return records

    def _load_layer(self, directory: Path, layer: DataLayer) -> List[Record]:
        """Load all records from a layer directory with error handling."""
        records = []

        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return records

        # Get all files
        jsonl_files = list(directory.glob("*.jsonl"))
        json_files = list(directory.glob("*.json"))
        all_files = jsonl_files + json_files

        # Progress bar
        file_iterator = tqdm(
            all_files,
            desc=f"  Loading {layer.value}",
            disable=not self.show_progress,
            unit="file",
        )

        for file_path in file_iterator:
            try:
                if file_path.suffix == ".jsonl":
                    layer_records = self._load_jsonl_file(file_path, layer)
                else:
                    layer_records = self._load_json_file(file_path, layer)

                records.extend(layer_records)

                if self.show_progress:
                    file_iterator.set_postfix(
                        {"records": len(records)}, refresh=False
                    )
            except Exception as e:
                error_info = {
                    "file": str(file_path),
                    "layer": layer.value,
                    "error": str(e),
                }
                self.errors.append(error_info)
                logger.warning(
                    f"Failed to load file: {file_path.name}",
                    extra=error_info,
                )
                # Continue processing other files

        return records

    def _load_jsonl_file(self, file_path: Path, layer: DataLayer) -> List[Record]:
        """Load records from a JSONL file with error handling."""
        records = []
        filename = file_path.name
        line_errors = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        record = Record.from_legacy_format(data)
                        record.layer = layer
                        record.origin_file = filename

                        if record.has_spanish() and record.has_translation_pair():
                            records.append(record)
                    except (json.JSONDecodeError, ValueError) as e:
                        line_errors += 1
                        if line_errors <= 5:  # Log first 5 errors
                            logger.debug(
                                f"Error in {filename}:{line_num}: {e}"
                            )
                        continue

            if line_errors > 0:
                logger.debug(
                    f"Loaded {filename} with {line_errors} line errors"
                )

        except Exception as e:
            raise DataLoadError(
                f"Failed to load JSONL file",
                file=str(file_path),
                error=str(e),
            )

        return records

    def _load_json_file(self, file_path: Path, layer: DataLayer) -> List[Record]:
        """Load records from a JSON dump file with error handling."""
        records = []
        filename = file_path.name

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = data.get('items', []) if isinstance(data, dict) else data

            for item in items:
                try:
                    if 'original' in item:
                        orig = item['original']
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
                except Exception:
                    continue

        except Exception as e:
            raise DataLoadError(
                f"Failed to load JSON file",
                file=str(file_path),
                error=str(e),
            )

        return records

    def _normalize_records(self, records: List[Record]) -> List[Record]:
        """Normalize all text fields with progress tracking."""
        from src.transforms.normalizers import normalize_record

        if self.show_progress:
            print("[3/5] Normalizing records...")

        normalized = []
        errors = 0

        record_iterator = tqdm(
            records,
            desc="  Normalizing",
            disable=not self.show_progress,
            unit="record",
        )

        for record in record_iterator:
            try:
                normalized_record = normalize_record(record, self.normalizer)
                normalized.append(normalized_record)
            except Exception as e:
                errors += 1
                if errors <= 10:
                    logger.debug(f"Normalization error: {e}")
                continue

        if errors > 0:
            logger.warning(f"Failed to normalize {errors} records")

        return normalized

    def _split_records(
        self,
        records: List[Record],
    ) -> Tuple[List[Record], List[Record], List[Record]]:
        """Split records into train/val/test sets."""
        if self.show_progress:
            print("[4/5] Splitting into train/val/test...")

        settings = get_settings()
        random.seed(self.seed)

        indices = list(range(len(records)))
        random.shuffle(indices)

        total = len(records)
        train_end = int(total * settings.train_ratio)
        val_end = int(total * (settings.train_ratio + settings.val_ratio))

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
        if self.show_progress:
            print("[5/5] Saving to Gold layer...")

        self.gold_dir.mkdir(parents=True, exist_ok=True)

        splits = [
            (train, "train_v1.jsonl"),
            (val, "validation_v1.jsonl"),
            (test, "test_v1.jsonl"),
        ]

        for records, filename in tqdm(
            splits,
            desc="  Saving splits",
            disable=not self.show_progress,
            unit="split",
        ):
            path = self.gold_dir / filename
            self._save_jsonl(records, path)

    def _save_jsonl(self, records: List[Record], path: Path):
        """Save records to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                data = record.to_dict()
                clean_data = {k: v for k, v in data.items() if v is not None}
                f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(records)} records to {path.name}")


# CLI entry point
if __name__ == "__main__":
    from src.utils.logger import configure_root_logger

    configure_root_logger()

    pipeline = UnifiedPipeline(show_progress=True)
    stats = pipeline.run()

    print("\n" + "=" * 70)
    print("  Pipeline Complete")
    print("=" * 70)
    print(f"\nRun ID:              {stats['run_id']}")
    print(f"Total unique records: {stats['total_records']}")
    print(f"  Train:             {stats['train_count']}")
    print(f"  Validation:        {stats['val_count']}")
    print(f"  Test:              {stats['test_count']}")
    print(f"\nLanguage distribution:")
    print(f"  Nahuatl:           {stats['nahuatl_count']}")
    print(f"  Maya:              {stats['maya_count']}")
    print(f"\nDuration:            {stats['duration_seconds']:.2f}s")
    print(f"Errors:              {stats['errors_count']}")
    print("=" * 70)
