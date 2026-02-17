# 🏛️ CORC-NAH: Enterprise Data Pipeline para Lenguas Indígenas

> **Arquitectura ETL de Producción** | Corpus Multilingüe (Náhuatl/Maya/Español) | Medallion Architecture | Data Quality Automation

[![CI/CD Pipeline](https://github.com/saidmoreno808/nahuatl-data-pipeline/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/saidmoreno808/nahuatl-data-pipeline/actions)
[![codecov](https://codecov.io/gh/saidmoreno808/nahuatl-data-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/saidmoreno808/nahuatl-data-pipeline)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Scala 2.12](https://img.shields.io/badge/scala-2.12-red.svg)](https://www.scala-lang.org/)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.0-017CEE.svg)](https://airflow.apache.org/)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC.svg)](https://www.terraform.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Great Expectations](https://img.shields.io/badge/data%20quality-Great%20Expectations-green.svg)](https://greatexpectations.io/)
[![Tests](https://img.shields.io/badge/tests-116%20passing-brightgreen.svg)](tests/)
[![Last Commit](https://img.shields.io/github/last-commit/saidmoreno808/nahuatl-data-pipeline)](https://github.com/saidmoreno808/nahuatl-data-pipeline/commits/main)


---

## 🎯 Portfolio Data Engineering

Este repositorio demuestra competencias técnicas para **Data Engineer** en entornos enterprise:

| Competencia | Implementación | Evidencia |
|-------------|----------------|-----------|
| **ETL Modular** | Arquitectura Bronze/Silver/Diamond/Gold con separation of concerns | [`src/pipeline/`](src/pipeline/), 116 tests ✅ |
| **Data Quality** | Great Expectations (8 validaciones corpus-specific) | [`great_expectations/`](great_expectations/) |
| **CI/CD** | GitHub Actions multi-version + Jenkins declarativo | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **Orquestación** | Apache Airflow DAGs + Jenkinsfile automation | [`airflow_dags/`](airflow_dags/), [Jenkinsfile](Jenkinsfile) |
| **SQL Analítico** | Metadata store con queries de lineage/quality | [`sql/queries/*.sql`](sql/queries/) |
| **Observabilidad** | Structured logging (JSON) + metrics tracking | [`src/utils/logger.py`](src/utils/logger.py) |
| **Testing** | Unit + Integration + Parity tests (>80% coverage) | [`tests/`](tests/) |
| **Scala/Spark** | High-performance deduplication (2.5x faster than PySpark) | [`src/scala_examples/`](src/scala_examples/) |
| **Enterprise DBs** | Oracle, Teradata, Generic JDBC connectors | [`src/connectors/`](src/connectors/) |
| **IaC** | Terraform templates for AWS (S3, Glue, Athena) | [`terraform/`](terraform/) |
| **Documentation** | ADRs documenting architectural decisions | [`docs/adr/`](docs/adr/), [ARCHITECTURE.md](ARCHITECTURE.md) |

📌 **Tecnologías:** Python, Scala, SQL, Terraform, Docker, Airflow, Great Expectations, pytest, Spark

---


---

## 🚀 Quick Demo (3 minutos)

### Prerequisitos
```bash
# Verificar versión Python
python --version  # Debe ser 3.9, 3.10, o 3.11

# Docker (opcional, para testing completo)
docker --version
```

### Instalación Express
```bash
# 1. Clonar repositorio
git clone https://github.com/saidmoreno808/corc-nah-enterprise.git
cd corc-nah-enterprise

# 2. Setup automático (crea venv + instala deps)
make install

# 3. Validar instalación
make test
```

**Output esperado:**
```
======================= 116 passed, 15 skipped in 2.36s =======================
✅ Pipeline listo para ejecutar
```

### Ver Pipeline en Acción

#### Opción 1: ETL Refactorizado (Production-Ready)
```bash
# Ejecutar pipeline completo con CLI
python -m src.pipeline.cli run

# Ver progreso con barra de loading
# [1/5] Loading Silver layer...  ━━━━━━━━━━━━━━━━━━━━ 100%
# [2/5] Loading Diamond layer... ━━━━━━━━━━━━━━━━━━━━ 100%
# [3/5] Normalizing records...   ━━━━━━━━━━━━━━━━━━━━ 100%
# [4/5] Splitting into train/val/test...
# [5/5] Saving to Gold layer...  ━━━━━━━━━━━━━━━━━━━━ 100%
# ✅ Pipeline Complete (4.2s)
```

#### Opción 2: Validar Calidad de Datos
```bash
# Ejecutar suite Great Expectations
python scripts/run_quality_check.py data/gold/train_v1.jsonl

# Output:
# ✅ Schema validation PASSED
# ✅ Null checks PASSED
# ✅ Unicode preservation (macrons) PASSED (32% records with ā/ē/ī/ō/ū)
# ⚠️  Duplicate detection PASSED (98.7% unique - see ADR for rationale)
# ✅ Length constraints PASSED
# ✅ Source validation PASSED
# ✅ Volume sanity check PASSED (1.4M records)
#
# HTML Report: great_expectations/uncommitted/data_docs/local_site/index.html
```

#### Opción 3: Comparar con Pipeline Legacy (Shadow Mode)
```bash
# Ejecutar tests de paridad (valida que refactored == legacy)
pytest tests/integration/test_parity_with_legacy.py -v

# Ver métricas comparativas
make benchmark
```

---

## 🏗️ Decisiones de Arquitectura

Este proyecto NO es un prototipo académico, es un **caso de estudio de Data Engineering enterprise-grade**:

### Principios de Diseño

1. **Separation of Concerns**
   - **Extractors:** Abstracción de fuentes (HuggingFace, YouTube, PDFs)
   - **Transformers:** Normalización Unicode, deduplicación fuzzy (desacoplados)
   - **Loaders:** Writers con retry logic y checkpointing
   - **Benefit:** Cada componente es testeable en aislamiento

2. **Fail-Fast Philosophy**
   - **Bronze:** Validación de schema básica (JSON válido, encoding UTF-8)
   - **Silver:** Quality gates (nulls, duplicates, text length)
   - **Diamond:** Validaciones avanzadas (Unicode preservation, source catalog)
   - **Gold:** Final checks antes de entregar a ML team
   - **Benefit:** Errores detectados temprano, no propagan downstream

3. **Observability First**
   - **Structured Logging:** JSON format (compatible con ELK, CloudWatch)
   - **Trace IDs:** Tracking de registros individuales a través de capas
   - **Metrics:** Prometheus-ready (duration, throughput, error rates)
   - **Benefit:** Debugging de producción es trivial, no "black box"

4. **Test-Driven Development**
   - **Unit Tests:** 88 tests (transforms, utils, models)
   - **Integration Tests:** 24 tests (shadow mode, end-to-end)
   - **Parity Tests:** 15 tests (refactored vs legacy comparison)
   - **Coverage:** 80%+ (crítico para confianza en refactoring)
   - **Benefit:** Refactoring seguro, sin romper funcionalidad existente

5. **Documentation as Code**
   - **ADRs (Architecture Decision Records):** Documentan el "por qué", no solo el "qué"
   - **Inline Comments:** Explican trade-offs y limitaciones conocidas
   - **Type Hints:** Python 3.9+ annotations para autocomplete
   - **Benefit:** Onboarding de nuevos devs en horas, no días

### Trade-offs Conscientes

Este proyecto hace elecciones técnicas deliberadas, optimizando para **demo portfolio** vs **producción enterprise**:

- **SQLite over PostgreSQL:** Elimina dependencias externas en desarrollo; misma API que producción
- **Pandas over full PySpark:** Corpus actual (<1M registros) no justifica overhead de cluster; Spark disponible para Diamond layer
- **Local Docker over cloud deployment:** Demo reproducible en cualquier máquina; arquitectura AWS documentada en `terraform/`
- **Deduplicación exacta + fuzzy MinHash LSH:** Balance entre velocidad (exact) y cobertura dialectal (fuzzy para variantes como _kuali tonali_ vs _kuale tunal_)

### 3. Data Quality Validation

Using **Great Expectations** to enforce:
- Null rate < 10% for critical columns
- Duplicate rate < 5%
- Text length distributions (detect corrupted data)
- Unicode character preservation

```bash
great_expectations checkpoint run gold_dataset_validation
```

---

## 🧪 Testing Strategy

### Test Pyramid

```
        /\
       /  \        E2E (manual smoke tests)
      /    \
     /------\      Integration (15+ parity tests)
    /        \
   /----------\    Unit (50+ tests, >90% coverage)
```

### Running Tests

```bash
# All tests
make test

# Parity tests only (CRITICAL)
make test-parity

# Unit tests
make test-unit

# Coverage report
make coverage
```

### Parity Test Examples

```python
def test_record_count_parity(golden_stats, golden_train_df):
    """New pipeline must produce exact same record count."""
    expected = golden_stats['train']['total_records']
    actual = len(golden_train_df)
    assert actual == expected

def test_unicode_preservation(golden_stats, golden_train_df):
    """Macrons MUST be preserved (zero tolerance)."""
    text = ''.join(golden_train_df['nah'].dropna())
    assert 'ā' in text  # Macron must exist
    assert 'ē' in text
```

---

## 📈 Data Quality Metrics

Current dataset statistics (v1):

| Metric | Value |
|--------|-------|
| Total Records | ~250,000 |
| Náhuatl Pairs | ~180,000 (72%) |
| Maya Pairs | ~70,000 (28%) |
| Duplicate Rate | 2.3% |
| Null Rate (Spanish) | 0.1% |
| Null Rate (Indigenous) | 0.5% |
| Macron Preservation | 100% ✅ |

View full report:
```bash
cat benchmark/golden_stats.json | jq
```

---

## 🛠️ Development Workflow

### 1. Before Making Changes

```bash
# Ensure golden dataset exists
make golden

# Verify baseline passes
make parity
```

### 2. During Refactoring

```bash
# Format code
make format

# Run tests continuously
pytest-watch tests/

# Check coverage
make coverage
```

### 3. Before Committing

```bash
# Run all quality checks
make check

# Verify parity still passes
make parity
```

---

## 🏗️ Architecture Decisions

We document **WHY** decisions were made using ADRs (Architectural Decision Records):

- [ADR-001: Why SQLite for Metadata](docs/adr/001-why-sqlite.md)
- [ADR-002: Unicode Normalization Strategy](docs/adr/002-unicode-normalization.md)
- [ADR-003: When to Use Spark vs Pandas](docs/adr/003-spark-evaluation.md)

**Key Principle:** We prioritize **understanding** over deployment. The Spark example code demonstrates knowledge of distributed systems without requiring a cluster.

---

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      # CRITICAL: Run parity tests
      - name: Parity Tests
        run: make parity

      - name: Unit Tests
        run: make test-unit

      - name: Data Quality
        run: |
          great_expectations checkpoint run gold_validation
```

### Jenkinsfile (Template)

```groovy
// Declarative pipeline showing understanding of Jenkins
pipeline {
    agent any

    stages {
        stage('Setup') {
            steps {
                sh 'make install-dev'
            }
        }

        stage('Parity Tests') {
            steps {
                sh 'make parity'
            }
        }

        stage('Quality Gates') {
            steps {
                sh 'make check'
            }
        }
    }
}
```

---

## 📚 Documentation

- **Setup Guide:** [docs/setup-windows.md](docs/setup-windows.md)
- **Architecture:** [docs/architecture.md](docs/architecture.md)
- **ADRs:** [docs/adr/](docs/adr/)
- **API Docs:** [docs/api/](docs/api/)

---

## 🤝 Contributing

This project demonstrates professional software engineering practices:

1. **Never break the golden dataset** - Parity tests MUST pass
2. **Document decisions** - Write ADRs for architectural choices
3. **Type everything** - Use Python type hints + mypy
4. **Test everything** - Maintain >90% coverage

```bash
# Before submitting PR
make check
make parity
```

---

## 📊 Metrics & Observability

### Logging

Structured JSON logging for easy parsing:

```python
logger.info(
    "etl_step_completed",
    extra={
        "step": "deduplication",
        "records_before": 100000,
        "records_after": 95000,
        "duplicate_rate": 0.05,
        "duration_seconds": 12.5
    }
)
```

### Metrics

Track pipeline performance:

```bash
# View processing stats
sqlite3 logs/metrics.db "SELECT * FROM pipeline_runs ORDER BY timestamp DESC LIMIT 10"
```

---

## 🎓 Learning Resources

This project demonstrates understanding of:

- **Data Engineering:** Bronze/Silver/Gold lakehouse, ETL patterns
- **Data Quality:** Great Expectations, regression testing, unicode handling
- **Software Engineering:** Type hints, testing pyramid, CI/CD
- **Big Data:** When to use Spark vs Pandas (see [docs/adr/003-spark-evaluation.md](docs/adr/003-spark-evaluation.md))

**Related Technologies (shown via templates/examples):**
- Apache Spark (see `src/spark_examples/`)
- AWS services (see `docs/aws-architecture.md`)
- Jenkins (see `Jenkinsfile`)
- Control-M (see `docs/controlm-integration.md`)

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Data Sources:** HuggingFace (AmericasNLP, Flores), YouTube (The Náhuatl Channel), Bible.is
- **Náhuatl Language Resources:** INALI (Instituto Nacional de Lenguas Indígenas)
- **Inspiration:** [Py-Elotl](https://github.com/ElotlMX/py-elotl), [Axolotl NLP](https://github.com/axolotl-ai-cloud/axolotl)

---

## 📧 Contact

**Said Moreno** - Data Engineer
LinkedIn: [Said Moreno](https://linkedin.com/in/said-moreno)
Email: said.moreno@email.com

---

**Built with ❤️ for the Náhuatl and Maya communities**
