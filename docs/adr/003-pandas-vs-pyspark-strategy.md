# ADR 003: Pandas vs PySpark Strategy

**Status:** ✅ Accepted
**Date:** 2026-01-28
**Authors:** Said Moreno
**Deciders:** Data Engineering Lead
**Technical Review:** Ml Engineering Team

---

## Context

El pipeline CORC-NAH procesa ~5 GB de datos crudos (Bronze) que se reduce a ~1 GB en Gold tras cleaning/deduplicación. La decisión de usar **Pandas** vs **PySpark** afecta:

- **Developer Experience:** Velocidad de desarrollo, debugging, testing
- **Performance:** Tiempo de ejecución end-to-end del pipeline
- **Infraestructura:** Costos de compute, complejidad de deployment
- **Escalabilidad:** Capacidad de crecer a 10x, 100x volumen

### Dataset Actual (Enero 2026)

| Capa | Tamaño | Registros | Características |
|------|--------|-----------|-----------------|
| **Bronze** (raw) | ~5 GB | ~2.5M | JSON/JSONL heterogéneo, duplicados |
| **Silver** (cleaned) | ~2 GB | ~1.8M | Normalizado, sin nulls críticos |
| **Diamond** (validated) | ~1.2 GB | ~1.5M | Deduplicado, quality checks |
| **Gold** (training-ready) | ~800 MB | ~1.4M | Splits (train/val/test), parquet |

**Operaciones Computacionalmente Intensivas:**
1. **Deduplicación Fuzzy:** MinHash LSH para detectar variantes dialectales (~O(n²) naive)
2. **Language Detection:** Modelo sklearn sobre texto completo (CPU-bound)
3. **Unicode Normalization:** `unicodedata.normalize()` sobre millones de strings

### Alternativas Evaluadas

| Framework | Throughput (records/sec) | Memory Peak | Dev Time | Deployment Complexity |
|-----------|---------------------------|-------------|----------|----------------------|
| **Pandas (single-core)** | ~800 | 4 GB | Baseline | Trivial (pip install) |
| **Pandas + Dask** | ~2,500 | 6 GB | +30% | Media (scheduler config) |
| **PySpark (local[4])** | ~3,200 | 8 GB | +80% | Alta (JVM tuning) |
| **PySpark (EMR cluster)** | ~15,000 | 16 GB/node | +100% | Muy alta (IaC) |

---

## Decision

**Estrategia Híbrida:**

1. **ETL Principal → Pandas**
   - Lectura, normalización, transformaciones lineales
   - Simplicidad > Performance para <10 GB

2. **Operaciones O(n²) → PySpark (bajo demanda)**
   - Deduplicación fuzzy con MinHash
   - Joins masivos (si aparecen datasets de referencia grandes)

3. **Migration Trigger Documentado**
   - Migrar a PySpark full si dataset raw >20 GB O tiempo >30 min

---

## Rationale

### Por Qué Pandas para ETL Core

#### 1. Developer Experience Excepcional

**Pandas:**
```python
# Debugging trivial - inspect interactivo
df = pd.read_json("data.jsonl", lines=True)
df[df['nah'].str.contains('ā', na=False)].head()  # ← 1 línea, readable

# Profiling con 1 línea
df.describe()
df.info(memory_usage='deep')
```

**PySpark:**
```python
# Debugging requiere collect() → cuidado con OOM
df = spark.read.json("data.jsonl")
df.filter(df.nah.contains('ā')).show()  # ← No muestra nulls correctamente

# Profiling requiere acciones explícitas
df.count()  # Trigger evaluation
df.schema  # Inferencia puede ser incorrecta
```

**Impacto:** Con Pandas, resolver un bug toma **5 minutos**. Con Spark, **30 minutos** (lazy evaluation obscurece errores).

#### 2. Testing Simplificado

**Pandas:**
```python
@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"es": "Hola", "nah": "Piyali"},
        {"es": "Adiós", "nah": "Tlahual"},
    ])

def test_normalizer(sample_df):
    result = normalize_text(sample_df)
    assert "ā" not in result['nah'].iloc[0]  # ← Assertion directa
```

**PySpark:**
```python
@pytest.fixture
def spark_session():
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    yield spark
    spark.stop()  # ← Cleanup crítico (leak de memory)

def test_normalizer(spark_session):
    df = spark_session.createDataFrame([...])
    result = normalize_text(df).collect()  # ← collect() puede OOM
    assert "ā" not in result[0]['nah']
```

**Impacto:** Suite de 116 tests con Pandas: **2.3 segundos**. Con Spark: **~15 segundos** (startup overhead).

#### 3. Deployment Cero-Fricción

**Pandas:**
```bash
# CI/CD
pip install pandas  # 5 MB wheel, 2 segundos

# Producción (Docker)
FROM python:3.10-slim
RUN pip install pandas  # ← Imagen <500 MB
```

**PySpark:**
```bash
# CI/CD
pip install pyspark  # 200 MB, requiere Java 8+
apt-get install openjdk-11-jre-headless  # +200 MB

# Producción (Docker)
FROM openjdk:11-jre-slim
RUN pip install pyspark  # ← Imagen >1 GB
ENV SPARK_CLASSPATH=...  # 20 líneas de env vars
```

**Impacto:** Con Pandas, desplegar en Fargate/Cloud Run es trivial. Spark requiere tuning de JVM.

#### 4. Memory Footprint Predecible

**Pandas (5 GB dataset):**
- Memory peak: **~4 GB** (carga completa en RAM)
- Predecible: `df.memory_usage(deep=True).sum() * 1.5` (factor seguridad)

**PySpark (5 GB dataset):**
- Executor memory: configurar heap (`--executor-memory 4g`)
- Off-heap memory: configurar overheads (`spark.executor.memoryOverhead=1g`)
- Python worker memory: configurar (`spark.python.worker.memory=2g`)
- **TOTAL:** Necesita ~8 GB por executor para procesar 5 GB datos

**Impacto:** En GitHub Actions (7 GB RAM máx), Pandas corre. Spark requiere self-hosted runner.

### Por Qué PySpark para Operaciones O(n²)

#### 1. Deduplicación Fuzzy (MinHash LSH)

**Problema:**
- Comparar 1.5M registros náhuatl para detectar duplicados con typos/variantes
- Naive: O(n²) = 2.25 trillion comparaciones → imposible

**Solución PySpark:**
```python
from pyspark.ml.feature import MinHashLSH

# Hash locality-sensitive
minhash = MinHashLSH(inputCol="text_vector", outputCol="hashes", numHashTables=5)
model = minhash.fit(df)

# Encontrar duplicados en O(n log n) en vez de O(n²)
duplicates = model.approxSimilarityJoin(df, df, threshold=0.8)
```

**Performance:**
- Pandas (naive nested loop): **48 horas** (estimado)
- Pandas + datasketch: **4 horas** (single-core)
- PySpark + MinHashLSH: **12 minutos** (4 cores local) → **3 minutos** (EMR 10 nodes)

**Conclusión:** PySpark justificado SOLO para esta operación específica.

#### 2. Joins Masivos (Future-Proof)

**Escenario:** Si se integra diccionario de 500k entradas náhuatl para enriquecimiento:

```python
# Pandas - cargar todo en memoria
df_corpus = pd.read_json("corpus.jsonl", lines=True)  # 1.5M rows
df_dict = pd.read_json("dictionary.jsonl", lines=True)  # 500k rows
merged = df_corpus.merge(df_dict, left_on='nah', right_on='lemma', how='left')
# ↑ Memory spike: 4 GB corpus + 1.5 GB dict = 5.5 GB peak
```

```python
# PySpark - streaming join
df_corpus = spark.read.json("corpus.jsonl")
df_dict = spark.read.json("dictionary.jsonl")
merged = df_corpus.join(df_dict, df_corpus.nah == df_dict.lemma, "left")
# ↑ Spill to disk si excede memoria
```

**Trigger:** Si join causa OOM en máquina con 16 GB RAM → migrar a Spark.

---

## Consequences

### Positivas

1. **Onboarding Rápido**
   - Pandas es estándar en Data Science → 90% devs ya lo conocen
   - PySpark requiere comprensión de DAGs, particiones, shuffle

2. **Iteración Ágil**
   - Ciclo dev con Pandas: code → run → debug → **2 min**
   - Ciclo dev con Spark: code → submit → logs S3 → debug → **10 min**

3. **Costos Optimizados (Fase Demo)**
   - Pandas: laptop/CI gratuito
   - Spark: EMR cluster ~$5/hora × 2 horas/día = $300/mes

4. **Menos Código Boilerplate**
   - Pandas ETL: **~300 líneas** core logic
   - Spark equivalente: **~500 líneas** (session management, checkpoint handling)

### Negativas

1. **No Escalable >20 GB Sin Refactor**
   - Si corpus crece a 50 GB, Pandas requiere reescritura completa
   - Documentado en Migration Trigger (ver abajo)

2. **CPU-Bound en Operaciones Pesadas**
   - Normalización de 1.5M strings: **Pandas 8 min** vs **Spark 2 min** (4 cores)
   - Aceptable para job batch overnight

3. **No Aprovecha Cluster Si Disponible**
   - Si empresa ya tiene EMR/Databricks, infrautilizado con Pandas
   - **Contraargumento:** El corpus actual no justifica aún el overhead de un cluster completo

---

## Migration Trigger: Cuándo Migrar a PySpark Full

Migrar si **2+ condiciones** se cumplen:

| # | Condición | Threshold | Monitoreo |
|---|-----------|-----------|-----------|
| 1 | **Dataset Size** | Raw >20 GB O Gold >5 GB | `du -sh data/bronze/` |
| 2 | **Execution Time** | Pipeline end-to-end >30 minutos | `logs/metadata.db: duration_seconds` |
| 3 | **Memory OOM** | 2+ crashes por OOM en 32 GB RAM | CloudWatch logs, `/var/log/syslog` |
| 4 | **Concurrency** | Requiere procesamiento paralelo de N fuentes | Roadmap feature request |
| 5 | **Production Deploy** | Cliente requiere pipeline en EMR/Databricks | Contract requirement |

### Pre-Migration Checklist

Antes de migrar, optimizar Pandas:

```python
# 1. Usar tipos eficientes
df = df.astype({
    'es': 'string',  # vs object (ahorra 30% RAM)
    'source': 'category',  # vs string (ahorra 50% RAM)
})

# 2. Chunked reading
for chunk in pd.read_json("large.jsonl", lines=True, chunksize=10000):
    process(chunk)

# 3. Lazy evaluation con Dask (drop-in replacement)
import dask.dataframe as dd
df = dd.read_json("large.jsonl", blocksize="64MB")  # ← Zero code change
```

**Resultado esperado:** Estas optimizaciones extienden límite Pandas a ~50 GB antes de necesitar Spark.

### Migration Path (2 semanas)

**Semana 1: Preparation**
```bash
# Día 1-2: PySpark local proof-of-concept
pip install pyspark
python scripts/migrate_to_spark.py --dry-run

# Día 3-4: Benchmark local vs EMR
pytest tests/performance/test_spark_parity.py
# Validar que PySpark produce resultados idénticos

# Día 5: Terraform infrastructure
cd infrastructure/
terraform apply -target=module.emr_cluster
```

**Semana 2: Deployment**
```bash
# Día 6-8: Code migration
# Reemplazar pandas con pyspark.sql en src/pipeline/unify.py
git checkout -b feature/spark-migration

# Día 9: Integration testing
pytest tests/integration/ --use-spark

# Día 10: Production rollout
# Blue-green deployment: 10% traffic → Spark, 90% → Pandas
# Monitor metrics, rollback si latencia > SLA
```

**Costo estimado:** 80 horas dev + $500 AWS (testing EMR).

---

## Performance Benchmarks (Real Data)

Ejecutado en laptop (16 GB RAM, i7-8cores) con dataset 5 GB Bronze:

| Operation | Pandas | PySpark (local[4]) | Speedup |
|-----------|--------|-------------------|---------|
| Read JSONL | 28s | 35s | 0.8x (Spark overhead) |
| Filter nulls | 2s | 3s | 0.67x |
| Unicode normalize | 480s | 120s | **4x** ✅ |
| Deduplicate (exact) | 12s | 15s | 0.8x |
| Deduplicate (fuzzy MinHash) | 14,400s (4h) | 720s (12min) | **20x** ✅ |
| Write Parquet | 8s | 6s | 1.3x |
| **TOTAL (without fuzzy)** | **530s (8.8 min)** | **179s (3 min)** | 3x |

**Interpretación:**
- Para ETL normal (sin fuzzy dedup), Pandas es **suficientemente rápido** (<10 min)
- Fuzzy deduplication es el ÚNICO bottleneck que justifica Spark
- **Decisión:** Usar Pandas + llamar script PySpark solo para fuzzy dedup

---

## Hybrid Approach Implementation

**Arquitectura Actual:**

```
┌─────────────────────────────────────────────────────┐
│ Pandas ETL (src/pipeline/unify.py)                 │
│   ├─ Load Bronze JSONL                             │
│   ├─ Normalize (unicodedata)                       │
│   ├─ Filter nulls                                  │
│   ├─ CALL: scripts/spark_fuzzy_dedup.py  ← PySpark │
│   ├─ Split train/val/test                          │
│   └─ Save Gold Parquet                             │
└─────────────────────────────────────────────────────┘
```

**`scripts/spark_fuzzy_dedup.py`:**
```python
#!/usr/bin/env python3
"""PySpark-powered fuzzy deduplication (MinHash LSH)."""

from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSH, CountVectorizer

def fuzzy_dedup(input_path, output_path, threshold=0.85):
    spark = SparkSession.builder \
        .appName("FuzzyDedup") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Leer datos de Pandas (pickle o parquet intermedio)
    df = spark.read.parquet(input_path)

    # Vectorizar texto
    cv = CountVectorizer(inputCol="nah_tokens", outputCol="features")
    vectorized = cv.fit(df).transform(df)

    # MinHash LSH
    minhash = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = minhash.fit(vectorized)
    duplicates = model.approxSimilarityJoin(vectorized, vectorized, threshold)

    # Escribir IDs de duplicados para que Pandas los filtre
    duplicates.select("datasetA.id", "datasetB.id", "distCol") \
        .write.parquet(output_path, mode="overwrite")

    spark.stop()

if __name__ == "__main__":
    # Ejecutado desde Pandas con subprocess
    fuzzy_dedup("temp/silver.parquet", "temp/duplicates.parquet")
```

**Llamada desde Pandas:**
```python
import subprocess

# Guardar estado intermedio
df.to_parquet("temp/silver.parquet")

# Llamar PySpark para operación pesada
subprocess.run([
    "python", "scripts/spark_fuzzy_dedup.py",
    "--input", "temp/silver.parquet",
    "--output", "temp/duplicates.parquet"
], check=True)

# Leer resultados de PySpark
duplicates = pd.read_parquet("temp/duplicates.parquet")
df_deduped = df[~df['id'].isin(duplicates['datasetB.id'])]
```

**Ventaja:** Best of both worlds - simplicidad Pandas + power Spark donde importa.

---

## Alternative Considered: Polars

**Polars** es framework moderno (Rust-based) que promete "Pandas performance + Spark scalability":

| Métrica | Pandas | Polars | PySpark |
|---------|--------|--------|---------|
| Read 5 GB CSV | 28s | **12s** ✅ | 35s |
| Filter + Aggregate | 8s | **3s** ✅ | 10s |
| Memory usage | 5 GB | **3 GB** ✅ | 8 GB |
| Maturity | 15 años | 3 años | 12 años |
| Ecosystem | 50k+ libs | ~500 libs | 10k+ libs |

**Por qué NO Polars (aún):**
- **Madurez:** Polars 0.20.x aún tiene breaking changes frecuentes
- **Ecosystem:** No tiene equivalente a `sklearn`, `great_expectations`, `pandera`
- **Learning curve:** Requiere aprender nuevo API (lazy/eager evaluation)

**Reevaluar en:** 2027 cuando Polars alcance v1.0 stable.

---

## References

- [Pandas Best Practices](https://pandas.pydata.org/docs/user_guide/scale.html) - Official scaling guide
- [PySpark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [When to Use Dask vs Spark](https://docs.dask.org/en/stable/spark.html) - Comparative analysis
- [Polars vs Pandas Benchmark](https://pola.rs/posts/benchmarks/) - Independent benchmark
- Internal: `tests/performance/benchmark_pandas_vs_spark.py` - Local benchmarks
- Internal: `docs/performance_report_2026-01.md` - Production profiling

---

## Decision Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-01-15 | Initial: 100% Pandas | Dataset <5 GB |
| 2026-01-22 | Add: PySpark for fuzzy dedup | 4h → 12min speedup |
| 2026-01-28 | Document: Migration triggers | Anticipate scaling needs |
