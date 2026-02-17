# CORC-NAH Project Structure

```
corc_nah_colab_v2/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # GitHub Actions CI pipeline
│       ├── data-quality.yml          # Great Expectations validation
│       └── parity-check.yml          # Golden dataset regression tests
│
├── benchmark/                        # 🟡 Golden Dataset (Día 0)
│   ├── golden_train_v1.jsonl        # Reference dataset for parity tests
│   ├── golden_validation_v1.jsonl
│   ├── golden_test_v1.jsonl
│   ├── golden_stats.json            # Statistical baseline
│   ├── checksums.txt                # MD5 checksums
│   └── generate_stats.py            # Metrics computation script
│
├── config/                           # Configuration management
│   ├── __init__.py
│   ├── settings.py                  # Pydantic settings (12-factor)
│   ├── logging.yaml                 # Structured logging config
│   └── ge_suite.yaml                # Great Expectations suite
│
├── data/                             # Data lake structure
│   ├── bronze/                      # Raw ingestion (immutable)
│   │   ├── hf_datasets/            # HuggingFace downloads
│   │   ├── youtube_transcripts/    # YouTube API responses
│   │   └── pdfs/                   # Scanned documents
│   ├── silver/                      # Cleaned + normalized
│   │   ├── distilled/              # Gemini-processed
│   │   ├── harvested/              # YouTube extracted
│   │   └── dumps/                  # Legacy migrations
│   ├── diamond/                     # Human-validated + synthetic
│   │   ├── manual/                 # Expert translations
│   │   └── synthetic/              # Generated data
│   └── gold/                        # Training-ready splits
│       ├── train_v1.jsonl
│       ├── validation_v1.jsonl
│       └── test_v1.jsonl
│
├── docs/                             # Documentation
│   ├── setup-windows.md             # WSL2 setup guide
│   ├── architecture.md              # System design
│   ├── adr/                         # Architectural Decision Records
│   │   ├── 001-why-sqlite.md
│   │   ├── 002-unicode-normalization.md
│   │   └── 003-spark-evaluation.md
│   └── api/                         # API documentation
│
├── logs/                             # Application logs
│   ├── etl_runs/
│   └── validation_reports/
│
├── scripts/                          # 🔴 Legacy code (Shadow Mode)
│   ├── unify_datasets.py            # PRESERVE until parity = 100%
│   ├── scrape_youtube.py            # Reference implementation
│   └── ...                          # Other legacy scripts
│
├── sql/                              # Data warehouse queries
│   ├── schema.sql                   # SQLite schema for metadata
│   ├── views/
│   │   ├── quality_trends.sql
│   │   └── dialect_distribution.sql
│   └── queries/
│       └── data_lineage.sql
│
├── src/                              # 🟢 Refactored pipeline (Tier 1+)
│   ├── __init__.py
│   ├── cli.py                       # Click CLI interface
│   │
│   ├── connectors/                  # Source adapters
│   │   ├── __init__.py
│   │   ├── huggingface.py          # HF datasets connector
│   │   ├── youtube.py              # YouTube Data API v3
│   │   └── pdf.py                  # PyMuPDF extractor
│   │
│   ├── transforms/                  # ETL logic
│   │   ├── __init__.py
│   │   ├── normalizers.py          # Unicode normalization (CRÍTICO)
│   │   ├── deduplicators.py        # Fuzzy matching (CRÍTICO)
│   │   ├── dialect_detector.py     # Náhuatl variant detection
│   │   └── quality_filters.py      # Data validation rules
│   │
│   ├── jobs/                        # Orchestration
│   │   ├── __init__.py
│   │   ├── ingest_job.py           # Bronze → Silver
│   │   ├── transform_job.py        # Silver → Diamond
│   │   └── publish_job.py          # Diamond → Gold
│   │
│   ├── models/                      # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py              # Pydantic models
│   │   └── enums.py                # Language codes, sources
│   │
│   ├── utils/                       # Shared utilities
│   │   ├── __init__.py
│   │   ├── logger.py               # Structured JSON logging
│   │   ├── config.py               # Settings loader
│   │   ├── metrics.py              # Performance tracking
│   │   └── db.py                   # SQLite context manager
│   │
│   └── spark_examples/              # 🎓 Educational code (NO deployment)
│       ├── compare_pandas_vs_spark.py
│       ├── distributed_dedup.py
│       └── README.md               # "When to use Spark"
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   │
│   ├── unit/                        # Isolated component tests
│   │   ├── test_normalizers.py
│   │   ├── test_deduplicators.py
│   │   └── test_dialect_detector.py
│   │
│   ├── integration/                 # End-to-end tests
│   │   ├── test_parity_with_legacy.py  # 🔥 CRITICAL
│   │   ├── test_pipeline_e2e.py
│   │   └── test_data_quality.py
│   │
│   └── fixtures/                    # Test data
│       ├── sample_nahuatl.jsonl
│       └── sample_maya.jsonl
│
├── .devcontainer/                   # VS Code Dev Container
│   └── devcontainer.json
│
├── .editorconfig                    # Cross-IDE config (UTF-8, LF)
├── .gitattributes                   # Force LF line endings
├── .gitignore
├── .pre-commit-config.yaml          # Black, isort, mypy
│
├── Dockerfile                       # Lightweight Python 3.10 image
├── docker-compose.yml               # LocalStack + SQLite
│
├── Jenkinsfile                      # Declarative pipeline (template)
├── Makefile                         # Developer shortcuts
├── pyproject.toml                   # Poetry dependencies
├── pytest.ini                       # Pytest configuration
├── README.md                        # Main documentation
└── setup.py                         # Editable install

```

## Key Principles

### 🔴 Legacy (Shadow Mode)
- `scripts/` → Preserved until parity tests pass
- No modifications except documentation

### 🟡 Validation Layer
- `benchmark/` → Golden dataset for regression testing
- `tests/integration/test_parity_with_legacy.py` → Continuous validation

### 🟢 New Architecture
- `src/` → Refactored, typed, tested code
- Mirrors legacy behavior but with clean abstractions

### 🎓 Documentation-Driven
- `docs/adr/` → Explain WHY, not just WHAT
- `src/spark_examples/` → Demonstrate understanding without deployment
- SQL views → Show data warehouse knowledge

## Critical Files for Bluetab Application

1. **Technical Skills Showcase**
   - `src/transforms/normalizers.py` → Unicode handling expertise
   - `sql/schema.sql` → Data modeling
   - `Jenkinsfile` → CI/CD understanding
   - `src/spark_examples/compare_pandas_vs_spark.py` → Big Data awareness

2. **Software Engineering Practices**
   - `tests/integration/test_parity_with_legacy.py` → Regression testing
   - `.github/workflows/ci.yml` → Automation
   - `config/ge_suite.yaml` → Data quality
   - `docs/adr/` → Decision documentation

3. **Data Engineering Maturity**
   - `data/` lakehouse structure (Bronze/Silver/Gold)
   - `src/utils/db.py` → Metadata management
   - `benchmark/generate_stats.py` → Observability
