-- Quality Metrics Trend Analysis
-- Shows how data quality evolves over time
-- Useful for detecting regressions and tracking improvements

WITH daily_metrics AS (
    SELECT
        DATE(qm.measured_at) as date,
        qm.metric_name,
        qm.metric_value,
        qm.dataset_split,
        pr.pipeline_name,
        pr.records_output
    FROM quality_metrics qm
    JOIN pipeline_runs pr ON qm.run_id = pr.run_id
    WHERE pr.status = 'success'
      AND qm.measured_at >= DATE('now', '-30 days')  -- Last 30 days
),
aggregated AS (
    SELECT
        date,
        metric_name,
        dataset_split,
        AVG(metric_value) as avg_value,
        MIN(metric_value) as min_value,
        MAX(metric_value) as max_value,
        STDDEV(metric_value) as stddev_value,
        COUNT(*) as sample_count
    FROM daily_metrics
    GROUP BY date, metric_name, dataset_split
)
SELECT
    date,
    metric_name,
    dataset_split,
    ROUND(avg_value, 4) as avg_value,
    ROUND(min_value, 4) as min_value,
    ROUND(max_value, 4) as max_value,
    ROUND(stddev_value, 4) as stddev_value,
    sample_count,
    -- Calculate 7-day moving average
    ROUND(
        AVG(avg_value) OVER (
            PARTITION BY metric_name, dataset_split
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        4
    ) as ma7_avg_value,
    -- Trend indicator (compare to 7 days ago)
    CASE
        WHEN avg_value > LAG(avg_value, 7) OVER (
            PARTITION BY metric_name, dataset_split ORDER BY date
        ) THEN '↑ Increasing'
        WHEN avg_value < LAG(avg_value, 7) OVER (
            PARTITION BY metric_name, dataset_split ORDER BY date
        ) THEN '↓ Decreasing'
        ELSE '→ Stable'
    END as trend
FROM aggregated
ORDER BY date DESC, metric_name, dataset_split;
