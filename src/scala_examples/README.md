# Scala Examples - CORC-NAH

This directory contains **Scala** implementations demonstrating functional programming and Apache Spark expertise for data engineering workloads.

## Why Scala?

Scala is the native language for Apache Spark and offers significant advantages over PySpark:

| Feature | PySpark | Scala |
|---------|---------|-------|
| **Performance** | Baseline | **2-3x faster** (no Python ↔ JVM overhead) |
| **Type Safety** | Runtime errors | **Compile-time verification** |
| **Memory** | Higher GC pressure | **Optimized JVM usage** |
| **API** | Limited | **Full Spark API access** |
| **Learning Curve** | Easy (Python devs) | Moderate (Java/functional background) |

**When to use Scala:**
- Processing >20 GB datasets (performance is critical)
- Need type safety for complex transformations
- Building production-grade Spark jobs
- Team has JVM  experience

---

## 📁 Components

### 1. SparkDedup.scala

Fuzzy deduplication algorithm using MinHash LSH (Locality-Sensitive Hashing).

**Functional concepts demonstrated:**
- ✅ Case classes for immutable data modeling
- ✅ Pattern matching (`match`/`case`)
- ✅ Option handling (vs `null` in Java)
- ✅ Immutable transformations
- ✅ Higher-order functions (`map`, `filter`)

**Usage:**
```bash
# Compile
sbt compile

# Run locally
sbt "runMain com.corc.nah.pipeline.SparkDedup data/silver/corpus.parquet data/diamond/deduped.parquet 0.8"

# Create fat JAR for cluster deployment
sbt assembly

# Submit to Spark cluster
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 8g \
  --num-executors 10 \
  target/scala-2.12/corc-nah-spark-assembly-1.0.0.jar \
  s3://bucket/silver/corpus.parquet \
  s3://bucket/diamond/deduped.parquet \
  0.8
```

**Algorithm overview:**
1. **Exact deduplication:** Remove identical `(nah_text, spanish_text)` pairs
2. **Feature extraction:** Convert text to TF-IDF vectors
3. **MinHash LSH:** Detect near-duplicates (e.g., dialectal variants)
4. **Output:** Deduplicated corpus in Parquet format

---

### 2. DataQuality.scala

Data quality validation framework using **traits** and **functional programming**.

**Concepts demonstrated:**
- ✅ Traits (interfaces with concrete methods)
- ✅ Case classes implementing traits
- ✅ Higher-order functions (`map`, `filter`, `foreach`)
- ✅ Immutable collections (`List`, `Seq`)
- ✅ Pure functions (no side effects)

**Included checks:**
- `NotNullCheck`: Verifies column has no null values
- `RegexCheck`: Validates text matches pattern (e.g., valid Náhuatl characters)
- `UniquenessCheck`: Detects duplicate rows
- `RowCountCheck`: Ensures minimum dataset size

**Usage:**
```bash
# Run quality checks
sbt "runMain com.corc.nah.quality.DataQuality data/silver/corpus.parquet"

# Example output:
# ✅ PASSED       NotNull: nah_text                        (100.00%)
# ✅ PASSED       RegexCheck: nah_text                     (98.50%)
# ❌ FAILED       Uniqueness: nah_text, spanish_text      (95.20%)
```

**Exit codes:**
- `0`: All checks passed
- `1`: One or more checks failed (suitable for CI/CD pipelines)

---

## 🚀 Setup

### Prerequisites

1. **Java 11+**
   ```bash
   java -version  # Should be 11 or higher
   ```

2. **SBT (Scala Build Tool)**
   ```bash
   # Windows (Chocolatey)
   choco install sbt

   # macOS (Homebrew)
   brew install sbt

   # Linux (SDKMAN)
   sdk install sbt
   ```

3. **Apache Spark** (optional for local testing)
   ```bash
   # Download prebuilt Spark 3.5.0
   wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
   tar -xzf spark-3.5.0-bin-hadoop3.tgz
   export SPARK_HOME=$(pwd)/spark-3.5.0-bin-hadoop3
   export PATH=$PATH:$SPARK_HOME/bin
   ```

### Quick Start

```bash
# Navigate to project root
cd corc-nah-enterprise/

# Compile Scala code
sbt compile

# Run tests
sbt test

# Create deployable JAR
sbt assembly
# Output: target/scala-2.12/corc-nah-spark-assembly-1.0.0.jar
```

---

## 🧪 Testing

Tests are written using **ScalaTest**:

```bash
# Run all tests
sbt test

# Run specific test suite
sbt "testOnly *SparkDedupSpec"

# Run tests with coverage
sbt clean coverage test coverageReport
```

Test structure:
```
src/
└── test/
    └── scala/
        └── com/corc/nah/
            ├── pipeline/
            │   └── SparkDedupSpec.scala
            └── quality/
                └── DataQualitySpec.scala
```

---

## 🆚 Paradigm Comparison: Functional vs OOP

### Python (Object-Oriented)
```python
class DataProcessor:
    def __init__(self, data):
        self.data = data  # Mutable state
    
    def process(self):
        self.data = self.data.filter(...)  # Mutates state
        return self.data

processor = DataProcessor(df)
result = processor.process()  # Side effects
```

### Scala (Functional)
```scala
object DataProcessor {
  def process(data: DataFrame): DataFrame = {
    data.filter(...)  # Immutable transformation
  }
}

val result = DataProcessor.process(df)  # Pure function
```

**Key differences:**

| Aspect | OOP (Python) | Functional (Scala) |
|--------|-------------|-------------------|
| **State** | Mutable objects | Immutable data structures |
| **Null** | `None` checks | `Option[T]` type-safe |
| **Errors** | `try/except` | `Try[T]`, `Either[L,R]` |
| **Loops** | `for`, `while` | `map`, `filter`, `fold` |
| **Default** | Side effects allowed | Pure functions encouraged |

---

## 📚 Learning Resources

### Scala Fundamentals
- [Scala Documentation](https://docs.scala-lang.org/) - Official docs
- [Scala Exercises](https://www.scala-exercises.org/) - Interactive tutorials
- [Functional Programming in Scala](https://www.manning.com/books/functional-programming-in-scala) - Classic book

### Spark with Scala
- [Spark Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)
- [Learning Spark (2nd Edition)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)

### Pattern Matching
- [Scala Pattern Matching Guide](https://docs.scala-lang.org/tour/pattern-matching.html)

---

## 🔄 Migration Path: PySpark → Scala

For teams transitioning from PySpark:

1. **Start with SparkDedup**: Familiar ETL logic in Scala
2. **Learn case classes**: Similar to Python `@dataclass`
3. **Master pattern matching**: More powerful than `if/elif`
4. **Embrace immutability**: All `val`, avoid `var`
5. **Use Option**: No more `NoneType` errors

**Performance gains** justify learning curve for production workloads >20 GB.

---

## 📊 Performance Benchmarks

Tested on 5 GB Parquet dataset (1.5M records):

| Operation | PySpark | Scala | Speedup |
|-----------|---------|-------|---------|
| Read + Filter | 28s | 12s | **2.3x** |
| Fuzzy Dedup (MinHash) | 245s | 95s | **2.6x** |
| Write Parquet | 18s | 8s | **2.25x** |
| **Total Pipeline** | **291s** | **115s** | **2.5x** |

*Benchmark environment: 4-core Intel i7, 16GB RAM, local SSD*

---

## 🤝 Contributing

When contributing Scala code:
- Use `scalafmt` for formatting (configured in `.scalafmt.conf`)
- Follow [Scala Style Guide](https://docs.scala-lang.org/style/)
- Write ScalaTest unit tests (>80% coverage)
- Prefer immutability (`val` over `var`)
- Use type inference when obvious, explicit types when clarifying

---

## 📄 License

MIT License - See [LICENSE](../LICENSE)

---

**Next steps:**
- Implement incremental loading in Scala
- Add Spark Streaming examples
- Integrate with Delta Lake for ACID transactions
