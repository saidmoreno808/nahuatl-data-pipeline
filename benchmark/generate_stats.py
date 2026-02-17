"""
Golden Dataset Statistics Generator

Computes comprehensive metrics for the golden dataset to establish
a baseline for regression testing during refactoring.

Usage:
    python benchmark/generate_stats.py
    python benchmark/generate_stats.py --input data/gold/train_v1.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def compute_unicode_stats(text_series: pd.Series) -> Dict[str, Any]:
    """
    Analyzes Unicode character distribution in text.
    CRITICAL for Náhuatl: Macrons, saltillo, digraphs.
    """
    if text_series.empty:
        return {
            "unique_chars": [],
            "special_chars_present": {},
            "macron_count": 0,
            "saltillo_count": 0,
        }

    all_text = "".join(text_series.dropna())

    # Náhuatl-specific characters
    special_chars = {
        "macron_a": "ā",  # U+0101
        "macron_e": "ē",  # U+0113
        "macron_i": "ī",  # U+012B
        "macron_o": "ō",  # U+014D
        "macron_u": "ū",  # U+016B
        "saltillo": "h",  # Glottal stop (varies by orthography)
        "tl_digraph": "tl",  # Common digraph
        "tz_digraph": "tz",
        "kw_digraph": "kw",
    }

    special_chars_present = {
        key: char in all_text for key, char in special_chars.items()
    }

    # Count macrons
    macron_chars = ["ā", "ē", "ī", "ō", "ū", "Ā", "Ē", "Ī", "Ō", "Ū"]
    macron_count = sum(all_text.count(char) for char in macron_chars)

    # Count saltillo representations (h in Nahuatl context)
    saltillo_count = all_text.count("h")

    # Get sorted unique characters (for manual inspection)
    unique_chars = sorted(set(all_text))

    return {
        "unique_chars": unique_chars[:100],  # Limit to first 100 for readability
        "unique_char_count": len(unique_chars),
        "special_chars_present": special_chars_present,
        "macron_count": macron_count,
        "saltillo_count": saltillo_count,
    }


def compute_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes data quality metrics.
    """
    total_records = len(df)

    if total_records == 0:
        return {
            "error": "Empty dataset",
            "total_records": 0,
        }

    # Null rates
    null_counts = df.isnull().sum().to_dict()
    null_rates = {col: count / total_records for col, count in null_counts.items()}

    # Duplicate rate
    duplicate_count = df.duplicated().sum()
    duplicate_rate = duplicate_count / total_records

    # Text length statistics
    length_stats = {}
    for col in ["es", "nah", "myn"]:
        if col in df.columns:
            lengths = df[col].dropna().str.len()
            if not lengths.empty:
                length_stats[col] = {
                    "mean": float(lengths.mean()),
                    "median": float(lengths.median()),
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "std": float(lengths.std()),
                }

    return {
        "null_counts": null_counts,
        "null_rates": null_rates,
        "duplicate_count": int(duplicate_count),
        "duplicate_rate": float(duplicate_rate),
        "length_stats": length_stats,
    }


def compute_language_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes distribution of languages and sources.
    """
    distribution = {}

    # Language presence
    for lang in ["es", "nah", "myn"]:
        if lang in df.columns:
            distribution[f"{lang}_records"] = int(df[lang].notna().sum())
            distribution[f"{lang}_rate"] = float(df[lang].notna().sum() / len(df))

    # Source distribution
    if "source" in df.columns:
        source_counts = df["source"].value_counts().head(10).to_dict()
        distribution["top_sources"] = {
            str(k): int(v) for k, v in source_counts.items()
        }

    # Layer distribution (if present)
    if "layer" in df.columns:
        layer_counts = df["layer"].value_counts().to_dict()
        distribution["layer_distribution"] = {
            str(k): int(v) for k, v in layer_counts.items()
        }

    # Origin file distribution
    if "origin_file" in df.columns:
        origin_counts = df["origin_file"].value_counts().head(10).to_dict()
        distribution["top_origin_files"] = {
            str(k): int(v) for k, v in origin_counts.items()
        }

    return distribution


def compute_golden_stats(jsonl_path: Path) -> Dict[str, Any]:
    """
    Master function to compute all statistics for a golden dataset.
    """
    print(f"📊 Computing statistics for: {jsonl_path}")

    # Load dataset
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except Exception as e:
        return {
            "error": f"Failed to load {jsonl_path}: {e}",
            "total_records": 0,
        }

    if df.empty:
        return {
            "error": "Empty dataset after loading",
            "total_records": 0,
        }

    print(f"   Loaded {len(df)} records")

    # Compute metrics
    stats = {
        "file": str(jsonl_path),
        "total_records": len(df),
        "columns": list(df.columns),
    }

    # Volume metrics
    print("   Computing volume metrics...")
    stats["volume_metrics"] = {
        "total_records": len(df),
        "es_records": int(df["es"].notna().sum()) if "es" in df.columns else 0,
        "nah_records": int(df["nah"].notna().sum()) if "nah" in df.columns else 0,
        "myn_records": int(df["myn"].notna().sum()) if "myn" in df.columns else 0,
    }

    # Quality metrics
    print("   Computing quality metrics...")
    stats["quality_metrics"] = compute_quality_metrics(df)

    # Language distribution
    print("   Computing language distribution...")
    stats["language_distribution"] = compute_language_distribution(df)

    # Unicode analysis (CRITICAL for Náhuatl)
    print("   Analyzing Unicode characters...")
    if "nah" in df.columns:
        stats["unicode_stats_nah"] = compute_unicode_stats(df["nah"])
    if "myn" in df.columns:
        stats["unicode_stats_myn"] = compute_unicode_stats(df["myn"])

    print("✅ Statistics computation complete")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden dataset statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark/golden_train_v1.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/golden_stats.json",
        help="Path to output statistics JSON file",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to process",
    )

    args = parser.parse_args()

    # Process all splits
    all_stats = {}

    if args.input != parser.get_default("input"):
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Error: {input_path} does not exist", file=sys.stderr)
            sys.exit(1)

        stats = compute_golden_stats(input_path)
        all_stats["single_file"] = stats
    else:
        # Multi-split mode
        for split in args.splits:
            input_path = Path(f"benchmark/golden_{split}_v1.jsonl")

            if not input_path.exists():
                print(f"⚠️  Warning: {input_path} not found, skipping")
                continue

            stats = compute_golden_stats(input_path)
            all_stats[split] = stats

    # Save combined statistics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Statistics saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for split, stats in all_stats.items():
        if "error" in stats:
            print(f"{split}: ❌ {stats['error']}")
            continue

        total = stats["total_records"]
        nah_count = stats["volume_metrics"].get("nah_records", 0)
        myn_count = stats["volume_metrics"].get("myn_records", 0)

        print(f"\n{split.upper()}:")
        print(f"  Total records: {total:,}")
        print(f"  Náhuatl records: {nah_count:,} ({nah_count/total*100:.1f}%)")
        print(f"  Maya records: {myn_count:,} ({myn_count/total*100:.1f}%)")

        if "unicode_stats_nah" in stats:
            macron_count = stats["unicode_stats_nah"]["macron_count"]
            print(f"  Macrons detected: {macron_count:,}")

        dup_rate = stats["quality_metrics"]["duplicate_rate"]
        print(f"  Duplicate rate: {dup_rate*100:.2f}%")


if __name__ == "__main__":
    main()
