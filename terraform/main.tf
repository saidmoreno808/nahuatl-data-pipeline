# Terraform Configuration for CORC-NAH AWS Deployment
# Version: 1.0.0
# Provider: AWS
# Resources: S3, Glue, Athena, IAM

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Optional: Remote state backend (production best practice)
  # backend "s3" {
  #   bucket = "corc-nah-terraform-state"
  #   key    = "state/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "CORC-NAH"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "DataEngineering"
    }
  }
}

# =====================
# S3 DATA LAKE (Medallion Architecture)
# =====================

resource "aws_s3_bucket" "data_lake" {
  bucket = "corc-nah-data-lake-${var.environment}"

  tags = {
    Name        = "CORC-NAH Data Lake"
    Environment = var.environment
  }
}

# Enable versioning for data recovery
resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access (security best practice)
resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule: Delete Bronze layer after 90 days (saves cost)
resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    id     = "expire-bronze-layer"
    status = "Enabled"

    filter {
      prefix = "bronze/"
    }

    expiration {
      days = 90
    }
  }

  rule {
    id     = "transition-silver-to-ia"
    status = "Enabled"

    filter {
      prefix = "silver/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"  # Infrequent Access (cheaper)
    }
  }
}

# Medallion layer folders
resource "aws_s3_object" "bronze_folder" {
  bucket = aws_s3_bucket.data_lake.id
  key    = "bronze/"
  acl    = "private"
}

resource "aws_s3_object" "silver_folder" {
  bucket = aws_s3_bucket.data_lake.id
  key    = "silver/"
  acl    = "private"
}

resource "aws_s3_object" "diamond_folder" {
  bucket = aws_s3_bucket.data_lake.id
  key    = "diamond/"
  acl    = "private"
}

resource "aws_s3_object" "gold_folder" {
  bucket = aws_s3_bucket.data_lake.id
  key    = "gold/"
  acl    = "private"
}

# =====================
# GLUE DATA CATALOG
# =====================

resource "aws_glue_catalog_database" "corc_nah_db" {
  name        = "corc_nah_${var.environment}"
  description = "Glue Data Catalog for CORC-NAH linguistic corpus"

  location_uri = "s3://${aws_s3_bucket.data_lake.bucket}/"
}

# Glue Crawler for auto-discovery of Gold layer schema
resource "aws_glue_crawler" "gold_layer_crawler" {
  name          = "corc-nah-gold-crawler"
  database_name = aws_glue_catalog_database.corc_nah_db.name
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.data_lake.bucket}/gold/"
  }

  # Run daily at 04:00 UTC
  schedule = "cron(0 4 * * ? *)"

  schema_change_policy {
    update_behavior = "UPDATE_IN_DATABASE"
    delete_behavior = "LOG"
  }

  tags = {
    Name = "Gold Layer Crawler"
  }
}

# =====================
# ATHENA ANALYTICS
# =====================

resource "aws_athena_workgroup" "analytics" {
  name = "corc-nah-analytics"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.data_lake.bucket}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }

  tags = {
    Name = "CORC-NAH Analytics Workgroup"
  }
}

# =====================
# IAM ROLES & POLICIES
# =====================

# Glue service role
resource "aws_iam_role" "glue_role" {
  name = "corc-nah-glue-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AWS managed Glue policy
resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

# Custom policy for S3 access
resource "aws_iam_role_policy" "glue_s3_access" {
  name = "glue-s3-access"
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      }
    ]
  })
}

# ETL service role (for Airflow/Lambda)
resource "aws_iam_role" "etl_role" {
  name = "corc-nah-etl-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = ["lambda.amazonaws.com", "ec2.amazonaws.com"]
        }
      }
    ]
  })
}

# ETL S3 access policy
resource "aws_iam_role_policy" "etl_s3_access" {
  name = "etl-s3-access"
  role = aws_iam_role.etl_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      }
    ]
  })
}

# =====================
# OUTPUTS
# =====================

output "data_lake_bucket" {
  value       = aws_s3_bucket.data_lake.bucket
  description = "S3 bucket name for data lake"
}

output "glue_database" {
  value       = aws_glue_catalog_database.corc_nah_db.name
  description = "Glue catalog database name"
}

output "athena_workgroup" {
  value       = aws_athena_workgroup.analytics.name
  description = "Athena workgroup for analytics"
}

output "glue_role_arn" {
  value       = aws_iam_role.glue_role.arn
  description = "Glue service role ARN"
}

output "etl_role_arn" {
  value       = aws_iam_role.etl_role.arn
  description = "ETL service role ARN"
}
