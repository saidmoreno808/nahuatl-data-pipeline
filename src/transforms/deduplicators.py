"""
Deduplication

Removes duplicate records while preserving highest-quality versions.
Prioritizes Diamond layer over Silver over Bronze.

Example:
    >>> deduplicator = Deduplicator()
    >>> unique_records = deduplicator.deduplicate(records)
"""

from collections import defaultdict
from typing import List

from src.models.schemas import Record
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    Deduplicates records based on content similarity.

    Uses exact match on normalized text (case-insensitive, whitespace-collapsed).
    Preserves records from higher-priority layers (Diamond > Silver > Bronze).

    Example:
        >>> dedup = Deduplicator()
        >>> records = [
        ...     Record(es="Hola", nah="Piyali", layer="silver"),
        ...     Record(es="Hola", nah="Piyali", layer="diamond"),  # Duplicate but higher quality
        ... ]
        >>> unique = dedup.deduplicate(records)
        >>> len(unique)
        1
        >>> unique[0].layer
        'diamond'  # Kept the Diamond version
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize deduplicator.

        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive
        logger.debug(f"Deduplicator initialized (case_sensitive={case_sensitive})")

    def deduplicate(
        self,
        records: List[Record],
        keep: str = "last",
    ) -> List[Record]:
        """
        Remove duplicate records.

        Args:
            records: List of records to deduplicate
            keep: Which duplicate to keep ('first', 'last', or 'best')
                  - 'first': Keep first occurrence
                  - 'last': Keep last occurrence (default, matches legacy)
                  - 'best': Keep highest-quality (by layer priority)

        Returns:
            list: Deduplicated records

        Example:
            >>> dedup = Deduplicator()
            >>> records = [...]  # 1000 records with duplicates
            >>> unique = dedup.deduplicate(records, keep="best")
            >>> len(unique)
            950  # 50 duplicates removed
        """
        # Validate keep strategy
        if keep not in ["first", "last", "best"]:
            raise ValueError(
                f"Invalid keep strategy '{keep}'. "
                f"Must be 'first', 'last', or 'best'."
            )

        if not records:
            return []

        initial_count = len(records)
        logger.info(f"Starting deduplication of {initial_count} records")

        # Generate dedup keys
        keyed_records = [
            (self._get_dedup_key(record), record) for record in records
        ]

        # Group by dedup key
        groups = defaultdict(list)
        for key, record in keyed_records:
            groups[key].append(record)

        # Select which record to keep from each group
        unique_records = []
        duplicate_count = 0

        for key, group in groups.items():
            if len(group) == 1:
                # No duplicates
                unique_records.append(group[0])
            else:
                # Duplicates found
                duplicate_count += len(group) - 1

                if keep == "first":
                    chosen = group[0]
                elif keep == "last":
                    chosen = group[-1]
                else:  # keep == "best" (already validated)
                    chosen = self._select_best_record(group)

                unique_records.append(chosen)

                logger.debug(
                    f"Duplicate group (key={key[:50]}...): "
                    f"{len(group)} records, kept {chosen.layer}"
                )

        final_count = len(unique_records)
        duplicate_rate = duplicate_count / initial_count if initial_count > 0 else 0

        logger.info(
            f"Deduplication complete",
            extra={
                "initial_count": initial_count,
                "final_count": final_count,
                "duplicates_removed": duplicate_count,
                "duplicate_rate": round(duplicate_rate, 4),
            },
        )

        return unique_records

    def _get_dedup_key(self, record: Record) -> str:
        """
        Generate deduplication key for a record.

        Key is based on normalized text content (case-insensitive by default).

        Args:
            record: Record to generate key for

        Returns:
            str: Deduplication key
        """
        # Get text fields
        es = record.es or ""
        nah = record.nah or ""
        myn = record.myn or ""

        # Normalize
        if not self.case_sensitive:
            es = es.lower()
            nah = nah.lower()
            myn = myn.lower()

        # Strip whitespace
        es = es.strip()
        nah = nah.strip()
        myn = myn.strip()

        # Combine into key
        return f"{es}|{nah}|{myn}"

    def _select_best_record(self, records: List[Record]) -> Record:
        """
        Select the best record from a group of duplicates.

        Priority:
        1. Layer priority (Diamond > Silver > Bronze)
        2. Completeness (has more non-null fields)
        3. Length (longer text assumed to be more complete)
        4. First occurrence (tie-breaker)

        Args:
            records: List of duplicate records

        Returns:
            Record: Best record
        """
        if len(records) == 1:
            return records[0]

        # Import here to avoid circular dependency
        from src.models.enums import DataLayer

        # Sort by multiple criteria
        def score_record(record: Record) -> tuple:
            # Layer priority (higher is better)
            # Map layer to priority value
            layer_priorities = {
                DataLayer.BRONZE: 0,
                DataLayer.SILVER: 1,
                DataLayer.DIAMOND: 2,
                DataLayer.GOLD: 3,
            }

            # Get priority, handling both enum and string values
            if isinstance(record.layer, DataLayer):
                layer_priority = layer_priorities.get(record.layer, 0)
            else:
                # Handle string values
                try:
                    layer_enum = DataLayer(record.layer)
                    layer_priority = layer_priorities.get(layer_enum, 0)
                except (ValueError, KeyError):
                    layer_priority = 0

            # Completeness (count non-null fields)
            completeness = sum([
                record.has_spanish(),
                record.has_nahuatl(),
                record.has_maya(),
            ])

            # Total text length
            total_length = (
                len(record.es or "")
                + len(record.nah or "")
                + len(record.myn or "")
            )

            # Return tuple for sorting (higher values = better)
            return (layer_priority, completeness, total_length)

        # Sort records by score (descending)
        sorted_records = sorted(records, key=score_record, reverse=True)

        return sorted_records[0]

    def get_duplicate_stats(self, records: List[Record]) -> dict:
        """
        Get statistics about duplicates without removing them.

        Args:
            records: List of records to analyze

        Returns:
            dict: Statistics about duplicates

        Example:
            >>> dedup = Deduplicator()
            >>> stats = dedup.get_duplicate_stats(records)
            >>> print(f"Duplicate rate: {stats['duplicate_rate']:.2%}")
        """
        if not records:
            return {
                "total_records": 0,
                "unique_records": 0,
                "duplicate_records": 0,
                "duplicate_rate": 0.0,
                "duplicate_groups": 0,
            }

        # Group by dedup key
        groups = defaultdict(list)
        for record in records:
            key = self._get_dedup_key(record)
            groups[key].append(record)

        # Count duplicates
        unique_count = len(groups)
        duplicate_groups = sum(1 for group in groups.values() if len(group) > 1)
        duplicate_count = sum(
            len(group) - 1 for group in groups.values() if len(group) > 1
        )

        return {
            "total_records": len(records),
            "unique_records": unique_count,
            "duplicate_records": duplicate_count,
            "duplicate_rate": duplicate_count / len(records),
            "duplicate_groups": duplicate_groups,
            "largest_group_size": max(len(group) for group in groups.values()),
        }


def deduplicate_records(
    records: List[Record],
    keep: str = "best",
    case_sensitive: bool = False,
) -> List[Record]:
    """
    Convenience function for deduplication.

    Args:
        records: Records to deduplicate
        keep: Which duplicate to keep ('first', 'last', 'best')
        case_sensitive: Whether to use case-sensitive matching

    Returns:
        list: Deduplicated records

    Example:
        >>> records = [...]
        >>> unique = deduplicate_records(records, keep="best")
    """
    deduplicator = Deduplicator(case_sensitive=case_sensitive)
    return deduplicator.deduplicate(records, keep=keep)


# Example usage
if __name__ == "__main__":
    from src.models.enums import DataLayer, DataSource
    from src.utils.logger import configure_root_logger

    configure_root_logger()

    print("=== Deduplication Tests ===\n")

    # Test 1: Basic deduplication
    print("Test 1: Basic deduplication")
    records = [
        Record(
            es="Hola",
            nah="Piyali",
            source=DataSource.MANUAL,
            layer=DataLayer.SILVER,
        ),
        Record(
            es="Hola",
            nah="Piyali",
            source=DataSource.MANUAL,
            layer=DataLayer.DIAMOND,
        ),
        Record(
            es="Adiós",
            nah="Cualli tonalli",
            source=DataSource.MANUAL,
            layer=DataLayer.SILVER,
        ),
    ]

    dedup = Deduplicator()
    unique = dedup.deduplicate(records, keep="best")

    print(f"  Original: {len(records)} records")
    print(f"  Unique:   {len(unique)} records")
    print(f"  First unique record layer: {unique[0].layer}")

    # Test 2: Get stats without deduplication
    print("\nTest 2: Duplicate statistics")
    stats = dedup.get_duplicate_stats(records)
    print(f"  Total records:     {stats['total_records']}")
    print(f"  Unique records:    {stats['unique_records']}")
    print(f"  Duplicates:        {stats['duplicate_records']}")
    print(f"  Duplicate rate:    {stats['duplicate_rate']:.2%}")
    print(f"  Duplicate groups:  {stats['duplicate_groups']}")

    # Test 3: Case sensitivity
    print("\nTest 3: Case sensitivity")
    records_case = [
        Record(es="HOLA", nah="Piyali", source="test", layer="silver"),
        Record(es="hola", nah="Piyali", source="test", layer="diamond"),
    ]

    dedup_insensitive = Deduplicator(case_sensitive=False)
    dedup_sensitive = Deduplicator(case_sensitive=True)

    unique_insensitive = dedup_insensitive.deduplicate(records_case, keep="best")
    unique_sensitive = dedup_sensitive.deduplicate(records_case, keep="best")

    print(f"  Case-insensitive: {len(unique_insensitive)} unique")
    print(f"  Case-sensitive:   {len(unique_sensitive)} unique")

    # Test 4: Layer priority
    print("\nTest 4: Layer priority")
    records_priority = [
        Record(es="Test", nah="Test", source="test", layer=DataLayer.BRONZE),
        Record(es="Test", nah="Test", source="test", layer=DataLayer.SILVER),
        Record(es="Test", nah="Test", source="test", layer=DataLayer.DIAMOND),
    ]

    unique_priority = dedup.deduplicate(records_priority, keep="best")
    print(f"  3 duplicates with different layers")
    print(f"  Kept layer: {unique_priority[0].layer} (expected: diamond)")

    print("\n✓ Deduplication tests complete!")
