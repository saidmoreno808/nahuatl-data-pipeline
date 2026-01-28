#!/usr/bin/env python3
"""
Data Quality Check Runner

Ejecuta suite Great Expectations sobre corpus JSONL y genera report HTML.

Usage:
    python scripts/run_quality_check.py data/gold/train_v1.jsonl
    python scripts/run_quality_check.py data/gold/train_v1.jsonl --suite custom_suite

Exit Codes:
    0: Todas las validaciones pasaron
    1: Al menos una validación falló
    2: Error de ejecución (archivo no encontrado, etc.)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.checkpoint import SimpleCheckpoint
except ImportError:
    print("❌ ERROR: Great Expectations not installed")
    print("Run: pip install great-expectations")
    sys.exit(2)

# Colores ANSI para output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def load_jsonl(file_path: Path) -> pd.DataFrame:
    """
    Load JSONL file into DataFrame.

    Args:
        file_path: Path to JSONL file

    Returns:
        DataFrame with corpus data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or malformed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print_info(f"Loading data from: {file_path}")

    records = []
    line_count = 0
    error_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:
                    print_warning(f"Line {line_num}: JSON decode error - {e}")

    if not records:
        raise ValueError(f"No valid records found in {file_path}")

    df = pd.DataFrame(records)

    print_success(f"Loaded {len(df):,} records from {line_count:,} lines")
    if error_count > 0:
        print_warning(f"Skipped {error_count} malformed lines")

    return df


def setup_great_expectations(
    project_root: Path,
) -> gx.data_context.DataContext:
    """
    Initialize Great Expectations context.

    Args:
        project_root: Root directory of project

    Returns:
        Great Expectations DataContext
    """
    ge_dir = project_root / "great_expectations"

    # Crear estructura si no existe
    if not ge_dir.exists():
        print_info("Initializing Great Expectations directory structure...")
        ge_dir.mkdir(parents=True)
        (ge_dir / "expectations").mkdir(exist_ok=True)
        (ge_dir / "uncommitted").mkdir(exist_ok=True)
        (ge_dir / "plugins").mkdir(exist_ok=True)

    # Crear great_expectations.yml si no existe
    config_file = ge_dir / "great_expectations.yml"
    if not config_file.exists():
        print_info("Creating Great Expectations configuration...")
        config = {
            "config_version": 3.0,
            "datasources": {},
            "expectations_store_name": "expectations_store",
            "validations_store_name": "validations_store",
            "evaluation_parameter_store_name": "evaluation_parameter_store",
            "checkpoint_store_name": "checkpoint_store",
            "stores": {
                "expectations_store": {
                    "class_name": "ExpectationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "expectations/",
                    },
                },
                "validations_store": {
                    "class_name": "ValidationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "uncommitted/validations/",
                    },
                },
                "evaluation_parameter_store": {
                    "class_name": "EvaluationParameterStore",
                },
                "checkpoint_store": {
                    "class_name": "CheckpointStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "checkpoints/",
                    },
                },
            },
            "data_docs_sites": {
                "local_site": {
                    "class_name": "SiteBuilder",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "uncommitted/data_docs/local_site/",
                    },
                    "site_index_builder": {
                        "class_name": "DefaultSiteIndexBuilder",
                    },
                },
            },
            "anonymous_usage_statistics": {
                "enabled": False,
            },
        }

        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # Cargar contexto
    context = gx.get_context(context_root_dir=str(ge_dir))
    return context


def run_validation(
    df: pd.DataFrame,
    suite_name: str,
    context: gx.data_context.DataContext,
) -> Dict:
    """
    Execute Great Expectations validation suite.

    Args:
        df: DataFrame to validate
        suite_name: Name of expectation suite
        context: Great Expectations context

    Returns:
        Validation results dictionary

    Raises:
        ValueError: If suite not found
    """
    print_header(f"Running Validation Suite: {suite_name}")

    # Cargar expectation suite
    suite_path = Path(context.root_directory) / "expectations" / f"{suite_name}.json"
    if not suite_path.exists():
        raise ValueError(f"Expectation suite not found: {suite_path}")

    print_info(f"Loading suite from: {suite_path}")

    with open(suite_path, 'r') as f:
        suite_dict = json.load(f)

    suite = context.add_or_update_expectation_suite(
        expectation_suite_name=suite_name,
        expectations=suite_dict.get("expectations", []),
    )

    # Crear batch request
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="corpus_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_identifier"},
    )

    # Crear validador
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # Ejecutar validación
    print_info("Executing validation...")
    results = validator.validate()

    return results.to_json_dict()


def print_results(results: Dict) -> bool:
    """
    Print validation results in human-readable format.

    Args:
        results: Validation results from Great Expectations

    Returns:
        True if all validations passed, False otherwise
    """
    print_header("Validation Results")

    success = results.get("success", False)
    statistics = results.get("statistics", {})

    print(f"Overall Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Evaluated Expectations: {statistics.get('evaluated_expectations', 0)}")
    print(f"Successful Expectations: {statistics.get('successful_expectations', 0)}")
    print(f"Failed Expectations: {statistics.get('unsuccessful_expectations', 0)}")
    print(f"Success Rate: {statistics.get('success_percent', 0):.1f}%\n")

    # Detalles de cada expectation
    print_header("Individual Expectations")

    for idx, result in enumerate(results.get("results", []), 1):
        expectation_type = result.get("expectation_config", {}).get("expectation_type", "Unknown")
        expectation_success = result.get("success", False)

        # Extraer kwargs para mostrar qué se validó
        kwargs = result.get("expectation_config", {}).get("kwargs", {})
        column = kwargs.get("column", "N/A")

        # Status symbol
        symbol = "✅" if expectation_success else "❌"

        print(f"{idx}. {symbol} {expectation_type}")
        if column != "N/A":
            print(f"   Column: {column}")

        # Si falló, mostrar detalles
        if not expectation_success:
            observed = result.get("result", {}).get("observed_value", "N/A")
            print(f"   {Colors.FAIL}Observed: {observed}{Colors.ENDC}")

            # Mostrar mensaje de error si existe
            if "exception_info" in result:
                exception = result["exception_info"].get("exception_message", "")
                print(f"   {Colors.FAIL}Error: {exception}{Colors.ENDC}")

        print()  # Línea en blanco entre expectations

    return success


def generate_html_report(context: gx.data_context.DataContext) -> str:
    """
    Generate HTML data docs.

    Args:
        context: Great Expectations context

    Returns:
        Path to generated HTML report
    """
    print_info("Building Data Docs...")

    try:
        context.build_data_docs()
        docs_path = Path(context.root_directory) / "uncommitted" / "data_docs" / "local_site" / "index.html"

        if docs_path.exists():
            return str(docs_path.resolve())
        else:
            print_warning("HTML report generated but path not found")
            return "N/A"
    except Exception as e:
        print_warning(f"Failed to build HTML docs: {e}")
        return "N/A"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run Great Expectations data quality checks on corpus JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default suite on training data
  python scripts/run_quality_check.py data/gold/train_v1.jsonl

  # Run custom suite
  python scripts/run_quality_check.py data/gold/train_v1.jsonl --suite custom_suite

  # Specify project root
  python scripts/run_quality_check.py data.jsonl --root /path/to/project
        """,
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to JSONL file to validate",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="corc_nah_corpus_suite",
        help="Name of expectation suite to use (default: corc_nah_corpus_suite)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(args.root).resolve()
    file_path = Path(args.file)

    if not file_path.is_absolute():
        file_path = project_root / file_path

    print_header("CORC-NAH Data Quality Check")
    print(f"Project Root: {project_root}")
    print(f"Input File:   {file_path}")
    print(f"Suite:        {args.suite}")

    try:
        # 1. Load data
        df = load_jsonl(file_path)

        # 2. Setup Great Expectations
        context = setup_great_expectations(project_root)

        # 3. Add pandas datasource
        if "pandas_datasource" not in context.list_datasources():
            context.add_datasource(
                "pandas_datasource",
                class_name="Datasource",
                execution_engine={
                    "class_name": "PandasExecutionEngine",
                },
                data_connectors={
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"],
                    },
                },
            )

        # 4. Run validation
        results = run_validation(df, args.suite, context)

        # 5. Print results
        success = print_results(results)

        # 6. Generate HTML report
        html_path = generate_html_report(context)

        print_header("Summary")
        if success:
            print_success("All data quality checks PASSED")
        else:
            print_error("Some data quality checks FAILED")

        if html_path != "N/A":
            print_info(f"HTML Report: {html_path}")
            print(f"\n{Colors.OKCYAN}Open in browser:{Colors.ENDC}")
            print(f"  file://{html_path}\n")

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(2)
    except ValueError as e:
        print_error(str(e))
        sys.exit(2)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
