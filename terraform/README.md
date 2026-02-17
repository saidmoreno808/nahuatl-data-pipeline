# Terraform - Infrastructure as Code for CORC-NAH

This directory contains Terraform configurations to deploy the CORC-NAH data lake infrastructure on AWS.

---

## 🏗️ Infrastructure Components

### 1. S3 Data Lake
- **Medallion layers**: Bronze, Silver, Diamond, Gold
- **Versioning**: Enabled for data recovery
- **Encryption**: AES-256 at rest
- **Lifecycle policies**:
  - Bronze: Expire after 90 days
  - Silver: Transition to Infrequent Access after 30 days

### 2. AWS Glue
- **Data Catalog**: Schema discovery for Gold layer
- **Crawler**: Runs daily at 04:00 UTC
- **Auto-schema updates**: Handles schema evolution

### 3. Amazon Athena
- **Workgroup**: `corc-nah-analytics`
- **Query engine**: SQL analytics on Parquet files
- **Results**: Stored in S3 with encryption

### 4. IAM Roles
- **Glue Role**: Least-privilege access for crawlers
- **ETL Role**: For Airflow/Lambda data processing

---

## 💰 Cost Estimate

| Service | Usage (monthly) | Cost/month |
|---------|-----------------|------------|
| **S3 Standard** (10 GB) | First 50 TB | $0.23 |
| **S3 Infrequent Access** (Silver, 5 GB) | After 30 days | $0.06 |
| **Glue Crawler** (1 DPU-hour/day) | 30 runs × 0.44 $/DPU-hour | $13.20 |
| **Athena Queries** (~100 GB scanned) | $5 per TB scanned | $0.50 |
| **Data Transfer** (Negligible) | Internal AWS | $0.00 |
| **TOTAL** | | **~$14/month** |

*Estimate for small-scale deployment. Production costs scale with data volume.*

---

## 🚀 Deployment

### Prerequisites

1. **AWS Account** with admin access
2. **Terraform CLI** (v1.0+)
   ```bash
   # Install Terraform
   # Windows (Chocolatey)
   choco install terraform

   # macOS (Homebrew)
   brew tap hashicorp/tap
   brew install hashicorp/tap/terraform

   # Linux (Ubuntu/Debian)
   wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
   sudo apt update && sudo apt install terraform
   ```

3. **AWS Credentials**
   ```bash
   aws configure
   # AWS Access Key ID: <YOUR_KEY>
   # AWS Secret Access Key: <YOUR_SECRET>
   # Default region: us-east-1
   ```

### Step-by-Step Deployment

```bash
# 1. Navigate to terraform directory
cd terraform/

# 2. Initialize Terraform (downloads AWS provider)
terraform init

# 3. Validate configuration
terraform validate

# 4. Preview changes (DRY RUN)
terraform plan -var="environment=staging"

# 5. Apply changes (creates infrastructure)
terraform apply -var="environment=production"

# Type 'yes' to confirm

# 6. View outputs
terraform output
```

### Expected output:
```
data_lake_bucket = "corc-nah-data-lake-production"
glue_database = "corc_nah_production"
athena_workgroup = "corc-nah-analytics"
glue_role_arn = "arn:aws:iam::123456789:role/corc-nah-glue-role-production"
etl_role_arn = "arn:aws:iam::123456789:role/corc-nah-etl-role-production"
```

---

## 🧪 Testing

### Validate Configuration

```bash
# Check syntax
terraform fmt -check

# Validate resources
terraform validate

# Plan without applying
terraform plan -out=tfplan
```

### Test Deployment

```bash
# Deploy to dev environment
terraform apply -var="environment=dev"

# Upload test file
aws s3 cp test.parquet s3://corc-nah-data-lake-dev/gold/test.parquet

# Run Glue crawler manually
aws glue start-crawler --name corc-nah-gold-crawler

# Query with Athena
aws athena start-query-execution \
  --query-string "SELECT * FROM corc_nah_dev.gold LIMIT 10" \
  --work-group corc-nah-analytics
```

---

## 🔄 Updating Infrastructure

```bash
# Modify main.tf or variables.tf

# Preview changes
terraform plan

# Apply updates
terraform apply

# View state
terraform show
```

---

## 🗑️ Destroying Infrastructure

> **WARNING**: This will DELETE all data in S3 buckets!

```bash
# Preview what will be destroyed
terraform plan -destroy

# Destroy all resources
terraform destroy

# Confirm with 'yes'
```

---

## 📁 File Structure

```
terraform/
├── main.tf         # Main infrastructure definition
├── variables.tf    # Input variables
├── outputs.tf      # Output values (embedded in main.tf)
├── README.md       # This file
└── terraform.tfstate  # State file (DO NOT commit to Git!)
```

---

## 🔐 Security Best Practices

### 1. Remote State Backend

For production, use S3 backend:

```hcl
# Add to main.tf
terraform {
  backend "s3" {
    bucket = "corc-nah-terraform-state"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}
```

### 2. State Locking

Use DynamoDB for state locking (prevents concurrent modifications):

```bash
# Create DynamoDB table
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### 3. Secrets Management

**Never commit AWS credentials to Git!**

Options:
- Use AWS CLI profiles
- Use environment variables:
  ```bash
  export AWS_ACCESS_KEY_ID=your-key
  export AWS_SECRET_ACCESS_KEY=your-secret
  ```
- Use AWS IAM roles (for EC2/ECS deployments)

### 4. Least Privilege Access

- IAM roles use minimum required permissions
- S3 buckets block public access by default
- Encryption enabled for all storage

---

## 📊 Monitoring

### CloudWatch Metrics

View in AWS Console → CloudWatch:
- S3 bucket size
- Glue crawler runtime
- Athena query performance

### Terraform State

```bash
# List all resources
terraform state list

# Show specific resource
terraform state show aws_s3_bucket.data_lake

# Remove resource from state (advanced)
terraform state rm aws_s3_bucket.data_lake
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/terraform.yml
name: Terraform Deployment

on:
  push:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
      
      - name: Terraform Init
        run: cd terraform && terraform init
      
      - name: Terraform Plan
        run: cd terraform && terraform plan
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: cd terraform && terraform apply -auto-approve
```

---

## 📚 Resources

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/best-practices.html)
- [AWS Glue Documentation](https://docs.aws.amazon.com/glue/)
- [Terraform State Management](https://www.terraform.io/docs/language/state/index.html)

---

## 🐛 Troubleshooting

### Error: Bucket already exists

```
Error: creating S3 Bucket: BucketAlreadyExists
```

**Solution**: S3 bucket names are globally unique. Change environment suffix:
```bash
terraform apply -var="environment=prod2"
```

### Error: Insufficient permissions

```
Error: UnauthorizedOperation: You are not authorized to perform this operation
```

**Solution**: Ensure AWS credentials have required permissions:
- `s3:*`
- `glue:*`
- `athena:*`
- `iam:CreateRole`, `iam:AttachRolePolicy`

---

## 📄 License

MIT License - See [LICENSE](../LICENSE)
