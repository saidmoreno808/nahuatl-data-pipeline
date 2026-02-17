# CORC-NAH: Pipeline de Datos para Lenguas Indígenas

> Corpus Multilingüe (Náhuatl/Maya/Español) | Arquitectura Medallion | Fine-tuning de LLMs | Calidad de Datos Automatizada

[![CI/CD Pipeline](https://github.com/saidmoreno808/nahuatl-data-pipeline/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/saidmoreno808/nahuatl-data-pipeline/actions)
[![codecov](https://codecov.io/gh/saidmoreno808/nahuatl-data-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/saidmoreno808/nahuatl-data-pipeline)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Scala 2.12](https://img.shields.io/badge/scala-2.12-red.svg)](https://www.scala-lang.org/)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.0-017CEE.svg)](https://airflow.apache.org/)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC.svg)](https://www.terraform.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-116%20passing-brightgreen.svg)](tests/)

---

## Descripción

CORC-NAH es un pipeline ETL para construir y mantener un corpus paralelo español–náhuatl–maya, orientado al entrenamiento de modelos de traducción automática para lenguas indígenas de México.

El sistema ingesta datos desde múltiples fuentes (HuggingFace, YouTube, Bible.is), los normaliza y deduplica a través de una arquitectura Medallion, y genera conjuntos de entrenamiento en formato JSONL/Parquet para fine-tuning de LLMs. El modelo base entrenado es Qwen3-4B con SFT y pares DPO generados con Gemini 2.5 Flash como modelo maestro.

**Stack principal:** Python · Scala · SQL · Terraform · Docker · Airflow · Great Expectations · pytest · Spark

---

## Contenido del repositorio

| Módulo | Descripción |
|--------|-------------|
| [`src/pipeline/`](src/pipeline/) | Pipeline ETL principal (Bronze → Silver → Diamond → Gold) |
| [`pipeline/`](pipeline/) | Módulos de ingesta, procesamiento y validación (versión legacy) |
| [`scripts/`](scripts/) | Scripts de recolección, minería y evaluación de datos |
| [`tests/`](tests/) | Suite de pruebas: unitarias, integración y paridad (116 tests) |
| [`airflow_dags/`](airflow_dags/) | DAGs de orquestación con Apache Airflow |
| [`terraform/`](terraform/) | Infraestructura como código para AWS (S3, Glue, Athena) |
| [`sql/`](sql/) | Esquema de la base de datos de metadatos y queries de linaje |
| [`src/scala_examples/`](src/scala_examples/) | Jobs Spark en Scala para deduplicación de alto rendimiento |
| [`src/connectors/`](src/connectors/) | Conectores Oracle, Teradata y JDBC genérico |
| [`docs/adr/`](docs/adr/) | Registros de decisiones de arquitectura (ADRs) |
| [`benchmark/`](benchmark/) | Datasets de evaluación y scripts de benchmarking |

---

## Instalación

### Requisitos

```bash
python --version  # 3.9, 3.10 o 3.11
docker --version  # opcional, para pruebas de integración completas
```

### Configuración

```bash
# 1. Clonar el repositorio
git clone https://github.com/saidmoreno808/nahuatl-data-pipeline.git
cd nahuatl-data-pipeline

# 2. Instalar dependencias
make install

# 3. Verificar instalación
make test
```

Salida esperada:

```
======================= 116 passed, 15 skipped in 2.36s =======================
```

---

## Uso

### Ejecutar el pipeline ETL completo

```bash
python -m src.pipeline.cli run
```

```
[1/5] Cargando capa Silver...   ━━━━━━━━━━━━━━━━━━━━ 100%
[2/5] Cargando capa Diamond...  ━━━━━━━━━━━━━━━━━━━━ 100%
[3/5] Normalizando registros... ━━━━━━━━━━━━━━━━━━━━ 100%
[4/5] Particionando train/val/test...
[5/5] Guardando capa Gold...    ━━━━━━━━━━━━━━━━━━━━ 100%
Pipeline completado (4.2s)
```

### Validar calidad de datos

```bash
python scripts/run_quality_check.py data/gold/train_v1.jsonl
```

```
Validación de esquema:        OK
Valores nulos:                OK
Preservación Unicode (macros): OK  (32% de registros con ā/ē/ī/ō/ū)
Detección de duplicados:      OK  (98.7% únicos)
Restricciones de longitud:    OK
Validación de fuentes:        OK
Verificación de volumen:      OK  (1.4M registros)

Reporte HTML: great_expectations/uncommitted/data_docs/local_site/index.html
```

### Pruebas de paridad (pipeline nuevo vs. legacy)

```bash
pytest tests/integration/test_parity_with_legacy.py -v
make benchmark
```

---

## Arquitectura

### Capas del pipeline (Medallion)

```
Fuentes externas
  ├── HuggingFace (AmericasNLP, Flores, Tatoeba)
  ├── YouTube API (subtítulos en náhuatl)
  └── Bible.is (traducciones paralelas)
          │
          ▼
    [Bronze]  Ingesta cruda — Parquet, append-only, sin modificaciones
          │
          ▼
    [Silver]  Normalización Unicode NFC, limpieza de texto, deduplicación exacta
          │
          ▼
    [Diamond] Deduplicación fuzzy con MinHash LSH (Spark), prioridad por fuente
          │
          ▼
    [Gold]    Particiones train/val/test en JSONL para entrenamiento de LLMs
```

### Principios de diseño

**Separación de responsabilidades**
- Extractores, transformadores y cargadores desacoplados; cada componente es testeable de forma independiente.

**Detección temprana de errores (Fail-Fast)**
- Cada capa aplica sus propias validaciones antes de propagar datos a la siguiente.
- Bronze valida schema y encoding; Silver controla nulos y longitudes; Diamond verifica preservación Unicode y catálogo de fuentes.

**Observabilidad**
- Logging estructurado en JSON (compatible con ELK y CloudWatch).
- Trace IDs por registro para seguimiento entre capas.
- Métricas de duración, throughput y tasa de error disponibles via SQLite.

**Decisiones de arquitectura documentadas**
- Las decisiones técnicas relevantes están registradas como ADRs en [`docs/adr/`](docs/adr/).
- Ver [ARCHITECTURE.md](ARCHITECTURE.md) para el diagrama completo del sistema.

### Decisiones técnicas principales

- **SQLite en lugar de PostgreSQL:** elimina dependencias externas en local; misma interfaz API.
- **Pandas para ETL general, Spark para Diamond:** el tamaño actual del corpus (<1M registros) no justifica el overhead de un cluster completo. Spark se usa exclusivamente en la capa de deduplicación fuzzy.
- **Scala Spark para deduplicación:** 2.5x más rápido que PySpark en el mismo job (benchmark: `src/scala_examples/`).
- **Deduplicación exacta + fuzzy MinHash LSH:** la deduplicación exacta cubre el 95% de los casos; LSH captura variantes dialectales (_kuali tonali_ vs. _kuale tunal_).

---

## Modelo de lenguaje

El corpus generado por este pipeline se usa para el fine-tuning del modelo Qwen3-4B:

- **SFT** sobre 60,059 pares españo–náhuatl consolidados (deduplicados).
- **DPO** con 7,928 pares de preferencia generados con Gemini 2.5 Flash como modelo maestro (score medio del chosen: 7.99/10).
- **Evaluación:** BLEU 33.97 sobre 1,000 muestras de prueba. Resultados en [`benchmark_metrics_qwen3_4b_v5.json`](benchmark_metrics_qwen3_4b_v5.json).

Scripts de entrenamiento: [`entrenamiento_qwen3_4b_v5.py`](entrenamiento_qwen3_4b_v5.py), [`kaggle_dpo_script.py`](kaggle_dpo_script.py).

---

## Pruebas

### Pirámide de tests

```
        /\
       /  \     E2E (smoke tests manuales)
      /    \
     /------\   Integración (paridad legacy vs. nuevo, shadow mode)
    /        \
   /----------\ Unitarias (transforms, utils, modelos)
```

### Ejecutar tests

```bash
make test          # todos los tests
make test-parity   # solo paridad (crítico antes de merge)
make test-unit     # solo unitarios
make coverage      # reporte de cobertura
```

### Ejemplos de tests de paridad

```python
def test_record_count_parity(golden_stats, golden_train_df):
    """El nuevo pipeline debe producir exactamente el mismo número de registros."""
    expected = golden_stats['train']['total_records']
    actual = len(golden_train_df)
    assert actual == expected

def test_unicode_preservation(golden_stats, golden_train_df):
    """Los macrónos deben preservarse sin excepción."""
    text = ''.join(golden_train_df['nah'].dropna())
    assert 'ā' in text
    assert 'ē' in text
```

---

## Métricas del corpus

| Métrica | Valor |
|---------|-------|
| Registros totales | ~250,000 |
| Pares español–náhuatl | ~180,000 (72%) |
| Pares español–maya | ~70,000 (28%) |
| Tasa de duplicados | 2.3% |
| Nulos (español) | 0.1% |
| Nulos (lengua indígena) | 0.5% |
| Preservación de macrónos | 100% |

```bash
cat benchmark/golden_stats.json | jq
```

---

## Flujo de desarrollo

### Antes de modificar

```bash
make golden   # generar dataset de referencia
make parity   # verificar línea base
```

### Durante el desarrollo

```bash
make format        # formatear código
pytest-watch tests/  # tests continuos
make coverage      # revisión de cobertura
```

### Antes de hacer commit

```bash
make check    # todas las verificaciones de calidad
make parity   # confirmar que paridad sigue pasando
```

---

## CI/CD

### GitHub Actions

```yaml
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
      - name: Tests de paridad
        run: make parity
      - name: Tests unitarios
        run: make test-unit
      - name: Calidad de datos
        run: great_expectations checkpoint run gold_validation
```

### Jenkinsfile

```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps { sh 'make install-dev' }
        }
        stage('Paridad') {
            steps { sh 'make parity' }
        }
        stage('Calidad') {
            steps { sh 'make check' }
        }
    }
}
```

---

## Documentación

- **Guía de configuración:** [docs/setup-windows.md](docs/setup-windows.md)
- **Arquitectura del sistema:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Decisiones de arquitectura (ADRs):** [docs/adr/](docs/adr/)

---

## Contribuir

1. Los tests de paridad deben pasar antes de cualquier merge.
2. Las decisiones de arquitectura relevantes deben documentarse como ADRs.
3. Usar type hints en todo el código Python.
4. Mantener cobertura de tests por encima del 80%.

```bash
make check
make parity
```

---

## Métricas y observabilidad

### Logging estructurado

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

### Consultar métricas de pipeline

```bash
sqlite3 logs/metrics.db "SELECT * FROM pipeline_runs ORDER BY timestamp DESC LIMIT 10"
```

---

## Licencia

MIT — ver [LICENSE](LICENSE)

---

## Fuentes de datos

- **HuggingFace:** AmericasNLP, Flores, Tatoeba, UniMorph
- **YouTube:** The Náhuatl Channel (subtítulos automáticos y manuales)
- **Bible.is:** Traducciones paralelas de las Escrituras en náhuatl huasteco y variantes
- **INALI:** Instituto Nacional de Lenguas Indígenas

**Proyectos relacionados:** [Py-Elotl](https://github.com/ElotlMX/py-elotl) · [Axolotl NLP](https://github.com/axolotl-ai-cloud/axolotl)

---

## Contacto

**Said Moreno**
LinkedIn: [said-moreno](https://linkedin.com/in/said-moreno)
