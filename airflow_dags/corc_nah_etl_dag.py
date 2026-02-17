"""
Airflow DAG for CORC-NAH ETL Pipeline

This DAG orchestrates the complete Medallion architecture pipeline:
Bronze → Silver → Diamond → Gold

Demonstrates enterprise orchestration patterns:
- Task dependencies (directed acyclic graph)
- SLA monitoring
- Error handling and retries
- Email notifications
- Parallel execution
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# Default arguments for tasks
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,  # Each run is independent
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'sla': timedelta(hours=2),  # Pipeline must complete in 2 hours
}

# Python callable functions for PythonOperator

def run_silver_pipeline(**context):
    """Execute Silver layer normalization."""
    logger.info("Starting Silver layer processing")
    # Import here to avoid DAG parsing delays
    from pipeline.layers.silver import SilverLayer
    
    silver = SilverLayer()
    result = silver.process()
    
    logger.info(f"Silver layer complete: {result['records_processed']} records")
    return result


def create_train_split(**context):
    """Create training split."""
    logger.info("Creating training data split")
    from pipeline.layers.gold import GoldLayer
    
    gold = GoldLayer()
    result = gold.create_split('train', ratio=0.8)
    
    logger.info(f"Train split: {result['record_count']} records")
    return result


def create_val_split(**context):
    """Create validation split."""
    logger.info("Creating validation data split")
    from pipeline.layers.gold import GoldLayer
    
    gold = GoldLayer()
    result = gold.create_split('validation', ratio=0.1)
    
    logger.info(f"Validation split: {result['record_count']} records")
    return result


def create_test_split(**context):
    """Create test split."""
    logger.info("Creating test data split")
    from pipeline.layers.gold import GoldLayer
    
    gold = GoldLayer()
    result = gold.create_split('test', ratio=0.1)
    
    logger.info(f"Test split: {result['record_count']} records")
    return result


def run_great_expectations(**context):
    """Run data quality validations."""
    logger.info("Running Great Expectations validations")
    import great_expectations as gx
    
    context_gx = gx.get_context()
    
    # Run checkpoint for Gold layer
    checkpoint_result = context_gx.run_checkpoint(
        checkpoint_name="gold_layer_checkpoint"
    )
    
    if not checkpoint_result.success:
        raise ValueError("Data quality checks failed!")
    
    logger.info("✅ All quality checks passed")
    return {"success": True, "suite": "gold_layer"}


# Define DAG
with DAG(
    'corc_nah_etl_pipeline',
    default_args=default_args,
    description='CORC-NAH ETL Pipeline (Bronze → Silver → Diamond → Gold)',
    schedule_interval='0 2 * * *',  # Daily at 02:00 UTC
    start_date=datetime(2026, 2, 1),
    catchup=False,  # Don't backfill historical runs
    tags=['data-engineering', 'etl', 'linguistics', 'nahuatl'],
    max_active_runs=1,  # Prevent concurrent pipeline runs
) as dag:

    # =====================
    # BRONZE LAYER
    # =====================
    
    bronze_ingestion = BashOperator(
        task_id='bronze_ingestion',
        bash_command='cd /opt/airflow && python -m src.pipeline.ingestion.manager',
        env={
            'PYTHONPATH': '/opt/airflow',
            'LOG_LEVEL': 'INFO'
        },
    )
    
    # =====================
    # SILVER LAYER
    # =====================
    
    silver_normalization = PythonOperator(
        task_id='silver_normalization',
        python_callable=run_silver_pipeline,
        provide_context=True,
    )
    
    # =====================
    # DIAMOND LAYER
    # =====================
    
    # Option 1: Use Python PySpark deduplication
    diamond_dedup_python = BashOperator(
        task_id='diamond_deduplication_pyspark',
        bash_command='cd /opt/airflow && python -m src.pipeline.processing.dedup',
    )
    
    # Option 2: Use Scala Spark job (requires spark-submit)
    # Uncomment to use Scala implementation instead:
    # diamond_dedup_scala = BashOperator(
    #     task_id='diamond_deduplication_scala',
    #     bash_command='''
    #         spark-submit \
    #         --master local[*] \
    #         --driver-memory 4g \
    #         --class com.corc.nah.pipeline.SparkDedup \
    #         /opt/airflow/target/scala-2.12/corc-nah-spark-assembly-1.0.0.jar \
    #         /opt/airflow/data/silver/corpus.parquet \
    #         /opt/airflow/data/diamond/deduped.parquet \
    #         0.8
    #     ''',
    # )
    
    # =====================
    # GOLD LAYER (Parallel Splits)
    # =====================
    
    gold_train = PythonOperator(
        task_id='gold_train_split',
        python_callable=create_train_split,
        provide_context=True,
    )
    
    gold_val = PythonOperator(
        task_id='gold_validation_split',
        python_callable=create_val_split,
        provide_context=True,
    )
    
    gold_test = PythonOperator(
        task_id='gold_test_split',
        python_callable=create_test_split,
        provide_context=True,
    )
    
    # =====================
    # QUALITY CHECKS
    # =====================
    
    quality_check = PythonOperator(
        task_id='quality_check_great_expectations',
        python_callable=run_great_expectations,
        provide_context=True,
    )
    
    # =====================
    # PUBLISH
    # =====================
    
    publish_s3 = BashOperator(
        task_id='publish_to_s3',
        bash_command='''
            # Sync Gold layer to S3 (requires AWS credentials)
            aws s3 sync /opt/airflow/data/gold/ s3://corc-nah-data/gold/ \
                --exclude "*.pyc" \
                --exclude "__pycache__/*" \
                --delete
            
            # Upload metadata
            aws s3 cp /opt/airflow/data/metadata/run_stats.json \
                s3://corc-nah-data/metadata/run_$(date +%Y%m%d).json
        ''',
        # Skip if AWS credentials not configured
        trigger_rule='none_failed',
    )
    
    # Optional: Publish to HuggingFace
    publish_huggingface = BashOperator(
        task_id='publish_to_huggingface',
        bash_command='cd /opt/airflow && python -m scripts.upload_to_hf',
        trigger_rule='all_done',  # Run even if S3 publish fails
    )
    
    # =====================
    # CLEANUP
    # =====================
    
    cleanup_temp = BashOperator(
        task_id='cleanup_temp_files',
        bash_command='''
            # Remove temporary files >7 days old
            find /opt/airflow/data/temp/ -type f -mtime +7 -delete
            
            # Clean old logs
            find /opt/airflow/logs/ -name "*.log" -mtime +30 -delete
        ''',
        trigger_rule='all_done',
    )
    
    # =====================
    # TASK DEPENDENCIES (DAG Structure)
    # =====================
    
    # Linear progression: Bronze → Silver → Diamond
    bronze_ingestion >> silver_normalization >> diamond_dedup_python
    
    # Parallel gold splits (all depend on diamond)
    diamond_dedup_python >> [gold_train, gold_val, gold_test]
    
    # Quality check (depends on all gold splits)
    [gold_train, gold_val, gold_test] >> quality_check
    
    # Publishing (sequential after quality check)
    quality_check >> publish_s3 >> publish_huggingface
    
    # Cleanup (after everything)
    publish_huggingface >> cleanup_temp


# =====================
# Additional DAG Metadata
# =====================

# This docstring appears in Airflow UI
dag.doc_md = """
# CORC-NAH ETL Pipeline

## Overview

This DAG implements the complete Medallion architecture for the CORC-NAH linguistic corpus:

1. **Bronze**: Raw data ingestion from YouTube, Bible.is, HuggingFace
2. **Silver**: Unicode normalization, text cleaning
3. **Diamond**: Fuzzy deduplication (MinHash LSH)
4. **Gold**: Train/Val/Test splits for ML training

## Schedule

- **Frequency**: Daily at 02:00 UTC
- **SLA**: 2 hours
- **Retries**: 3 attempts with exponential backoff

## Monitoring

- **Email alerts**: Sent on failure to data-team@company.com
- **Slack notifications**: (Configure via Airflow connections)
- **Metrics**: Stored in SQLite metadata DB

## Deployment

```bash
# Local development
docker-compose up airflow-webserver

# Production
kubectl apply -f k8s/airflow-deployment.yaml
```

## Contact

Data Engineering Team - data-team@company.com
"""
