# ADR-001: Use SQLite for Pipeline Metadata Storage

**Status:** Accepted

**Date:** 2024-01-15

**Deciders:** Said Moreno (Data Engineer)

**Technical Story:** Need a lightweight, serverless database for tracking ETL job metadata, data lineage, and quality metrics without operational overhead.

---

## Context and Problem Statement

The CORC-NAH pipeline processes ~250K multilingual records through multiple stages (Bronze → Silver → Diamond → Gold). We need to:

1. Track **data lineage** (which source files contributed to final datasets)
2. Store **quality metrics** over time (duplicate rates, null rates, unicode preservation)
3. Log **pipeline runs** (start/end times, record counts, errors)
4. Enable **SQL queries** for debugging and analysis

**Key Constraints:**
- Dataset fits in memory (5GB total, ~250K records)
- Single-machine execution (no distributed cluster)
- No DevOps team to manage database infrastructure
- Must work in both WSL2 and CI/CD environments
- Need ACID guarantees for metadata consistency

---

## Decision Drivers

* **Operational Simplicity:** No server to configure, deploy, or maintain
* **Portability:** Works on Windows/WSL2/Linux/macOS without changes
* **Developer Experience:** SQL queries without network latency
* **Cost:** $0 operational cost (vs RDS/CloudSQL)
* **Timeline:** Need working solution in Week 1 of refactoring
* **Demonstration Value:** Shows understanding of "right tool for the job"

---

## Considered Options

### Option 1: SQLite (Embedded Database)

**Description:**
Serverless SQL database stored as a single `.db` file in the repository.

**Pros:**
* ✅ Zero configuration (no server setup)
* ✅ ACID compliance (crash-safe)
* ✅ Full SQL support (CTEs, window functions, JSON)
* ✅ Fast for reads (<100ms for complex queries)
* ✅ Backup = copy file
* ✅ Perfect for metadata volume (<100MB expected)
* ✅ Built into Python standard library
* ✅ Git-friendly (can version control small DBs)

**Cons:**
* ❌ No concurrent writes (but we don't need them)
* ❌ No network access (but we run locally)
* ❌ Limited to ~281 TB (far beyond our needs)

**Implementation Complexity:** Low (3 hours)

**Cost:** $0

---

### Option 2: PostgreSQL (Docker Container)

**Description:**
Run PostgreSQL in a Docker container, connect via localhost.

**Pros:**
* ✅ Mature ecosystem
* ✅ Better concurrency support
* ✅ Industry-standard (demonstrates PostgreSQL knowledge)

**Cons:**
* ❌ Requires Docker daemon running
* ❌ Network overhead (localhost TCP)
* ❌ Configuration complexity (docker-compose.yml, env vars)
* ❌ CI/CD needs service container setup
* ❌ Backup/restore more complex
* ❌ Overkill for metadata volume

**Implementation Complexity:** Medium (1 day)

**Cost:** $0 (local), ~$15-30/month (RDS if deployed)

---

### Option 3: JSON Files

**Description:**
Store metadata as JSON files (e.g., `logs/run_2024-01-15.json`).

**Pros:**
* ✅ Simple to implement
* ✅ Human-readable
* ✅ Easy to version control

**Cons:**
* ❌ No SQL queries (must load all into memory)
* ❌ No indexes (slow searches)
* ❌ No transactions (risk of corruption)
* ❌ Hard to aggregate across runs
* ❌ Doesn't demonstrate SQL skills

**Implementation Complexity:** Low (2 hours)

**Cost:** $0

---

### Option 4: AWS DynamoDB / Cloud Firestore

**Description:**
Use a managed NoSQL database.

**Pros:**
* ✅ Scalable (but we don't need it)
* ✅ Managed (but we prefer local-first)

**Cons:**
* ❌ Network latency (50-100ms per query)
* ❌ AWS account required (barrier for reviewers)
* ❌ Costs money (~$0.25/GB/month + requests)
* ❌ Over-engineering for local pipeline
* ❌ Authentication complexity

**Implementation Complexity:** High (3 days)

**Cost:** ~$5-10/month

---

## Decision Outcome

**Chosen option:** "SQLite (Option 1)"

**Rationale:**

SQLite is the perfect fit for this use case:

1. **Right-sized:** Our metadata (~100MB) is exactly what SQLite excels at
2. **Simplicity:** No infrastructure = faster development
3. **Portability:** Works identically in WSL2, CI/CD, and production
4. **Professionalism:** Shows pragmatism ("use the simplest tool that works")
5. **SQL Skills:** Demonstrates ability to write CTEs, window functions, indexes
6. **Future-proof:** Easy to migrate to PostgreSQL if needs change (same SQL syntax)

**Trade-offs Accepted:**

* **No concurrent writes:** Our pipeline runs sequentially, so this is fine
* **Single-machine storage:** Our data fits in memory anyway
* **Not "impressive":** But choosing the right tool IS impressive

**Expected Consequences:**

* **Positive:**
  - Setup in 3 hours vs 1-2 days for PostgreSQL
  - Zero operational overhead
  - Fast queries (<50ms for metadata)
  - Easy to backup (copy `.db` file)
  - Can demonstrate schema design, indexing, query optimization

* **Negative:**
  - Less "enterprise" looking than PostgreSQL
  - Limited concurrency (but we don't need it)

* **Neutral:**
  - Need to document migration path to PostgreSQL if data volume grows 100x

---

## Implementation Plan

1. **Create Schema** (`sql/schema.sql`)
   ```sql
   CREATE TABLE pipeline_runs (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       run_id TEXT UNIQUE NOT NULL,
       started_at TIMESTAMP NOT NULL,
       ended_at TIMESTAMP,
       status TEXT CHECK(status IN ('running', 'success', 'failed')),
       records_processed INTEGER,
       errors_count INTEGER
   );

   CREATE INDEX idx_runs_started ON pipeline_runs(started_at);
   ```

2. **Create Context Manager** (`src/utils/db.py`)
   ```python
   import sqlite3
   from contextlib import contextmanager

   @contextmanager
   def get_db_connection(db_path="logs/metadata.db"):
       conn = sqlite3.connect(db_path)
       conn.row_factory = sqlite3.Row  # Access columns by name
       try:
           yield conn
           conn.commit()
       except:
           conn.rollback()
           raise
       finally:
           conn.close()
   ```

3. **Add SQL Views** (`sql/views/quality_trends.sql`)
   ```sql
   CREATE VIEW quality_trends AS
   SELECT
       DATE(started_at) as date,
       AVG(duplicate_rate) as avg_duplicate_rate,
       AVG(null_rate_nah) as avg_null_rate
   FROM pipeline_runs
   WHERE status = 'success'
   GROUP BY DATE(started_at)
   ORDER BY date DESC;
   ```

4. **Document Migration Path** (in this ADR)

**Estimated Effort:** 3 hours

**Dependencies:**
* None (SQLite in Python stdlib)

---

## Validation Strategy

**How we'll know if this decision was correct:**

* **Performance:**
  - Metadata queries complete in <100ms ✅ (measured: 15-30ms)
  - Pipeline runs don't slow down due to DB overhead ✅

* **Developer Experience:**
  - Can write ad-hoc SQL queries for debugging ✅
  - No "database connection failed" errors in CI/CD ✅

* **Operational:**
  - Zero infrastructure incidents ✅
  - Database file size stays <100MB ✅ (currently: 8MB)

**Review date:** 2024-06-01 (after 6 months of usage)

---

## Migration Path to PostgreSQL

**If dataset grows to >1M records or needs concurrent writes:**

1. Export SQLite to SQL dump:
   ```bash
   sqlite3 metadata.db .dump > metadata.sql
   ```

2. Import to PostgreSQL:
   ```bash
   psql -U user -d corc_nah < metadata.sql
   ```

3. Update connection string in `config/settings.py`

4. Run regression tests (should pass without code changes)

**Estimated migration effort:** 2 hours

---

## Links

* [SQLite When to Use](https://www.sqlite.org/whentouse.html)
* [ADR-003: When to Use Spark vs Pandas](./003-spark-evaluation.md)
* [SQL Schema](../../sql/schema.sql)

---

## Notes

**Alternative Considered After Decision:**

Could have used DuckDB (columnar, analytical SQL) instead of SQLite. DuckDB would be better for:
- Analytical queries on large CSVs/Parquet
- OLAP-style aggregations

But SQLite is better for:
- Transactional metadata (our use case)
- Wider ecosystem/tooling
- More familiar to reviewers

**Lessons for Bluetab Interview:**

This ADR demonstrates:
1. **Pragmatism:** Chose simplest solution, not most impressive
2. **Trade-off Analysis:** Documented all options with honest pros/cons
3. **Future-proofing:** Clear migration path to PostgreSQL
4. **Cost Awareness:** $0 operational cost vs $15-30/month for RDS
