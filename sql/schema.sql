-- CORC-NAH Metadata Database Schema
-- SQLite 3.x compatible
-- Version: 1.0.0

-- ============================================================================
-- Pipeline Execution Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,  -- UUID for idempotency
    pipeline_name TEXT NOT NULL,  -- 'ingest', 'transform', 'publish'
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT NOT NULL CHECK(status IN ('running', 'success', 'failed', 'cancelled')),
    records_input INTEGER,
    records_output INTEGER,
    records_filtered INTEGER,
    duration_seconds REAL,
    error_message TEXT,
    git_commit_hash TEXT,  -- Reproducibility
    config_snapshot TEXT   -- JSON snapshot of configuration
);

CREATE INDEX idx_runs_started ON pipeline_runs(started_at);
CREATE INDEX idx_runs_status ON pipeline_runs(status);
CREATE INDEX idx_runs_pipeline ON pipeline_runs(pipeline_name, started_at);

-- ============================================================================
-- Data Quality Metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,  -- 'duplicate_rate', 'null_rate_nah', etc.
    metric_value REAL NOT NULL,
    metric_unit TEXT,  -- 'percentage', 'count', 'seconds'
    dataset_split TEXT CHECK(dataset_split IN ('train', 'validation', 'test', 'all')),
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_metrics_run ON quality_metrics(run_id);
CREATE INDEX idx_metrics_name ON quality_metrics(metric_name, measured_at);

-- ============================================================================
-- Data Lineage Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_lineage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source_file TEXT NOT NULL,  -- Input file path
    source_type TEXT NOT NULL,  -- 'hf_dataset', 'youtube_transcript', 'pdf'
    source_layer TEXT NOT NULL CHECK(source_layer IN ('bronze', 'silver', 'diamond')),
    records_ingested INTEGER NOT NULL,
    records_valid INTEGER NOT NULL,
    records_invalid INTEGER NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT,  -- MD5 or SHA256
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_lineage_run ON data_lineage(run_id);
CREATE INDEX idx_lineage_source ON data_lineage(source_file);
CREATE INDEX idx_lineage_layer ON data_lineage(source_layer, ingested_at);

-- ============================================================================
-- Unicode Character Tracking (CRITICAL for Náhuatl)
-- ============================================================================

CREATE TABLE IF NOT EXISTS unicode_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    language TEXT NOT NULL CHECK(language IN ('nah', 'myn', 'es')),
    char_type TEXT NOT NULL,  -- 'macron', 'saltillo', 'digraph'
    char_value TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    dataset_split TEXT CHECK(dataset_split IN ('train', 'validation', 'test', 'all')),
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_unicode_run ON unicode_stats(run_id);
CREATE INDEX idx_unicode_lang ON unicode_stats(language, char_type);

-- ============================================================================
-- Validation Errors Log
-- ============================================================================

CREATE TABLE IF NOT EXISTS validation_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    record_id TEXT,  -- Identifier of problematic record
    error_type TEXT NOT NULL,  -- 'missing_translation', 'invalid_unicode', etc.
    error_severity TEXT NOT NULL CHECK(error_severity IN ('error', 'warning', 'info')),
    error_message TEXT NOT NULL,
    record_context TEXT,  -- JSON snapshot of problematic record
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_errors_run ON validation_errors(run_id);
CREATE INDEX idx_errors_type ON validation_errors(error_type, detected_at);
CREATE INDEX idx_errors_severity ON validation_errors(error_severity);

-- ============================================================================
-- Deduplication Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS deduplication_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    duplicate_key TEXT NOT NULL,  -- Hash of (es + nah + myn)
    occurrence_count INTEGER NOT NULL,  -- How many times this combo appeared
    kept_from_layer TEXT CHECK(kept_from_layer IN ('silver', 'diamond')),
    discarded_count INTEGER NOT NULL,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_dedup_run ON deduplication_log(run_id);
CREATE INDEX idx_dedup_key ON deduplication_log(duplicate_key);

-- ============================================================================
-- Dataset Statistics Snapshots
-- ============================================================================

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    dataset_split TEXT NOT NULL CHECK(dataset_split IN ('train', 'validation', 'test')),
    total_records INTEGER NOT NULL,
    nah_records INTEGER NOT NULL,
    myn_records INTEGER NOT NULL,
    es_records INTEGER NOT NULL,
    avg_nah_length REAL,
    avg_myn_length REAL,
    avg_es_length REAL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX idx_snapshots_run ON dataset_snapshots(run_id);
CREATE INDEX idx_snapshots_split ON dataset_snapshots(dataset_split, created_at);

-- ============================================================================
-- System Configuration Audit
-- ============================================================================

CREATE TABLE IF NOT EXISTS config_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config_key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT,  -- 'manual', 'automated', 'git_pull'
    change_reason TEXT
);

CREATE INDEX idx_config_audit_key ON config_audit(config_key, changed_at);

-- ============================================================================
-- Materialized Views (CTEs for complex queries)
-- ============================================================================

-- Note: SQLite doesn't support materialized views, but we can create
-- regular views and periodically refresh a cache table if needed

CREATE VIEW IF NOT EXISTS latest_quality_metrics AS
SELECT
    qm.metric_name,
    qm.metric_value,
    qm.metric_unit,
    qm.dataset_split,
    qm.measured_at,
    pr.pipeline_name,
    pr.git_commit_hash
FROM quality_metrics qm
JOIN pipeline_runs pr ON qm.run_id = pr.run_id
WHERE pr.status = 'success'
  AND pr.started_at = (
      SELECT MAX(started_at)
      FROM pipeline_runs
      WHERE status = 'success'
  );

CREATE VIEW IF NOT EXISTS quality_trends AS
SELECT
    DATE(qm.measured_at) as date,
    qm.metric_name,
    AVG(qm.metric_value) as avg_value,
    MIN(qm.metric_value) as min_value,
    MAX(qm.metric_value) as max_value,
    COUNT(*) as sample_count
FROM quality_metrics qm
JOIN pipeline_runs pr ON qm.run_id = pr.run_id
WHERE pr.status = 'success'
GROUP BY DATE(qm.measured_at), qm.metric_name
ORDER BY date DESC, qm.metric_name;

CREATE VIEW IF NOT EXISTS pipeline_performance AS
SELECT
    pipeline_name,
    DATE(started_at) as date,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
    AVG(duration_seconds) as avg_duration_seconds,
    AVG(records_output) as avg_records_output
FROM pipeline_runs
GROUP BY pipeline_name, DATE(started_at)
ORDER BY date DESC, pipeline_name;

CREATE VIEW IF NOT EXISTS data_lineage_summary AS
SELECT
    dl.source_layer,
    dl.source_type,
    COUNT(DISTINCT dl.source_file) as unique_sources,
    SUM(dl.records_ingested) as total_records_ingested,
    SUM(dl.records_valid) as total_records_valid,
    ROUND(
        100.0 * SUM(dl.records_valid) / NULLIF(SUM(dl.records_ingested), 0),
        2
    ) as validity_rate_pct
FROM data_lineage dl
JOIN pipeline_runs pr ON dl.run_id = pr.run_id
WHERE pr.status = 'success'
GROUP BY dl.source_layer, dl.source_type
ORDER BY dl.source_layer, total_records_ingested DESC;

-- ============================================================================
-- Triggers for Data Integrity
-- ============================================================================

-- Automatically calculate duration when pipeline completes
CREATE TRIGGER IF NOT EXISTS calculate_duration
AFTER UPDATE OF ended_at ON pipeline_runs
WHEN NEW.ended_at IS NOT NULL AND OLD.ended_at IS NULL
BEGIN
    UPDATE pipeline_runs
    SET duration_seconds = (
        julianday(NEW.ended_at) - julianday(NEW.started_at)
    ) * 86400
    WHERE id = NEW.id;
END;

-- ============================================================================
-- Sample Queries Documentation
-- ============================================================================

-- See sql/queries/ directory for complex analytical queries:
-- - quality_trends.sql: Track metrics over time
-- - data_lineage.sql: Trace records to source files
-- - pipeline_health.sql: Success rates and performance
-- - unicode_preservation.sql: Verify macron counts

-- ============================================================================
-- Schema Version
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES
    ('1.0.0', 'Initial schema with pipeline runs, quality metrics, and data lineage');
