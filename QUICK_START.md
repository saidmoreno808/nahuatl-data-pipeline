# 🚀 CORC-NAH Quick Start Guide

**Time to First Success:** ~30 minutes

---

## Prerequisites Checklist

Before starting, verify you have:

```
[ ] Windows 11 with WSL2 Ubuntu 22.04 installed
[ ] Docker Desktop with WSL2 integration enabled
[ ] Python 3.10+ installed in WSL2
[ ] Git configured in WSL2
[ ] 16GB RAM available
[ ] ~10GB free disk space
```

**Not sure?** Run this command in WSL2:
```bash
curl -sSL https://raw.githubusercontent.com/youruser/corc-nah/main/scripts/check_prereqs.sh | bash
```

---

## 5-Minute Setup

### 1. Clone & Enter (WSL2 Terminal)

```bash
# ⚠️ MUST be in WSL2, NOT Windows
cd ~
mkdir -p projects
cd projects
git clone https://github.com/youruser/corc-nah.git
cd corc-nah
```

### 2. Environment Setup

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies (takes ~3 minutes)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Verify Environment

```bash
python tests/validate_environment.py
```

**Expected output:**
```
✓ Python 3.10.x detected
✓ Virtual environment active
✓ All required packages installed
✓ Git configured with LF line endings
✓ UTF-8 encoding enabled
✓ Docker accessible
✓ WSL2 native filesystem detected
✓ Project structure complete

✅ 9/9 checks passed - Environment is ready!
```

---

## Day 0: Generate Golden Dataset

### Step 1: Create Baseline

```bash
make golden
```

This will:
1. Run legacy pipeline (`scripts/unify_datasets.py`)
2. Copy outputs to `benchmark/`
3. Generate checksums
4. Compute statistics

**Time:** ~2-5 minutes (depending on data volume)

**Expected output:**
```
📊 Generating golden dataset...
   Processing JSONL files...
   Processing JSON dumps...
   Found X datasets in Silver layer...
   Found Y datasets in Diamond layer...
   Total raw records loaded: 250,000
   Deduplication removed 5,000 records. Final count: 245,000
   Split sizes: Train=220,500, Val=12,250, Test=12,250
✅ Golden dataset generated

📊 Computing statistics...
   Loaded 220,500 records
   Computing volume metrics...
   Computing quality metrics...
   Computing language distribution...
   Analyzing Unicode characters...
✅ Statistics computation complete

📁 Statistics saved to: benchmark/golden_stats.json
```

### Step 2: Verify Checksums

```bash
md5sum -c benchmark/checksums.txt
```

**Expected output:**
```
benchmark/golden_train_v1.jsonl: OK
benchmark/golden_validation_v1.jsonl: OK
benchmark/golden_test_v1.jsonl: OK
```

### Step 3: Run Parity Tests

```bash
make parity
```

**Expected output:**
```
tests/integration/test_parity_with_legacy.py::test_golden_dataset_exists PASSED
tests/integration/test_parity_with_legacy.py::test_record_count_parity PASSED
tests/integration/test_parity_with_legacy.py::test_language_distribution_parity PASSED
tests/integration/test_parity_with_legacy.py::test_duplicate_rate_parity PASSED
tests/integration/test_parity_with_legacy.py::test_null_rate_parity PASSED
tests/integration/test_parity_with_legacy.py::test_unicode_preservation PASSED
tests/integration/test_parity_with_legacy.py::test_macron_count_parity PASSED
...
====== 15 passed in 3.45s ======
```

---

## Explore the Data

### View Statistics

```bash
# Pretty-print JSON
cat benchmark/golden_stats.json | jq

# Or use Python
python -c "
import json
stats = json.load(open('benchmark/golden_stats.json'))
print(f\"Total records: {stats['train']['total_records']:,}\")
print(f\"Náhuatl: {stats['train']['volume_metrics']['nah_records']:,}\")
print(f\"Maya: {stats['train']['volume_metrics']['myn_records']:,}\")
print(f\"Duplicate rate: {stats['train']['quality_metrics']['duplicate_rate']*100:.2f}%\")
print(f\"Macrons detected: {stats['train']['unicode_stats_nah']['macron_count']:,}\")
"
```

### Inspect Sample Records

```bash
# First 3 records from training set
head -n 3 benchmark/golden_train_v1.jsonl | jq
```

**Example output:**
```json
{
  "es": "Buenos días",
  "nah": "Cualli tonalli",
  "layer": "diamond",
  "origin_file": "manual_translations.jsonl"
}
{
  "es": "¿Cómo estás?",
  "nah": "¿Quēnin timotlaneltoquia?",
  "layer": "silver",
  "origin_file": "youtube_harvested.jsonl"
}
{
  "es": "Gracias",
  "nah": "Tlazohcamati",
  "layer": "diamond",
  "origin_file": "bible_aligned.jsonl"
}
```

### Check for Unicode Characters

```bash
# Count macrons in Náhuatl text
python -c "
import json
text = ' '.join([
    json.loads(line)['nah']
    for line in open('benchmark/golden_train_v1.jsonl')
    if 'nah' in json.loads(line)
])
macrons = ['ā', 'ē', 'ī', 'ō', 'ū']
for m in macrons:
    count = text.count(m)
    print(f'{m}: {count:,} occurrences')
"
```

---

## Development Workflow

### Run All Tests

```bash
make test
```

### Run Specific Test Categories

```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Parity tests only
make test-parity
```

### Check Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run all quality checks
make check
```

### Watch Tests (Continuous)

```bash
# Auto-run tests on file changes
pytest-watch tests/
```

---

## Common Commands

### Full Quality Check (Before Committing)

```bash
make check
# Runs: format → lint → test → parity
```

### View SQL Schema

```bash
sqlite3 logs/metadata.db < sql/schema.sql
sqlite3 logs/metadata.db ".schema"
```

### Generate Coverage Report

```bash
make coverage
open htmlcov/index.html  # Or use browser
```

### Clean Generated Files

```bash
# Remove cache files
make clean

# Remove everything including venv
make clean-all
```

---

## Troubleshooting

### ❌ "Permission denied" on scripts

```bash
chmod +x scripts/*.sh
git update-index --chmod=+x scripts/*.sh
```

### ❌ "Command not found: make"

```bash
sudo apt install make
```

### ❌ Line ending errors (^M characters)

```bash
# Fix all Python files
find . -type f -name "*.py" | xargs dos2unix

# Or re-clone in WSL2
cd ~/projects
rm -rf corc-nah
git clone https://github.com/youruser/corc-nah.git
```

### ❌ "ModuleNotFoundError"

```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### ❌ "Golden dataset not found"

```bash
# Generate it first
make golden
```

### ❌ Docker not accessible

```bash
# Check Docker Desktop is running
docker --version

# Enable WSL2 integration
# Docker Desktop → Settings → Resources → WSL Integration
# Enable "Ubuntu-22.04"
```

---

## Next Steps

Once Day 0 is complete (✅ golden dataset generated, ✅ parity tests passing):

1. **Read Architecture:** [docs/architecture.md](docs/architecture.md)
2. **Read ADRs:** [docs/adr/](docs/adr/)
3. **Start Refactoring:** Follow [DAY_0_SUMMARY.md](DAY_0_SUMMARY.md) → Week 1 plan
4. **Watch this:** [Video Tutorial](https://youtube.com/watch?v=example) (if available)

---

## Interactive Demo

Try this interactive exploration:

```bash
# 1. Count records by source layer
python -c "
import json
from collections import Counter
layers = [
    json.loads(line).get('layer')
    for line in open('benchmark/golden_train_v1.jsonl')
]
for layer, count in Counter(layers).most_common():
    print(f'{layer}: {count:,} records')
"

# 2. Find longest Náhuatl sentence
python -c "
import json
records = [json.loads(line) for line in open('benchmark/golden_train_v1.jsonl')]
longest = max(
    (r for r in records if 'nah' in r),
    key=lambda r: len(r['nah'])
)
print(f\"ES: {longest['es']}\")
print(f\"NAH: {longest['nah']}\")
print(f\"Length: {len(longest['nah'])} chars\")
"

# 3. Check dialect distribution (if available)
python -c "
import json
from collections import Counter
# Placeholder: implement dialect detection
print('Dialect detection not yet implemented')
"
```

---

## Getting Help

- **Environment Issues:** `python tests/validate_environment.py`
- **Setup Guide:** [docs/setup-windows.md](docs/setup-windows.md)
- **Project Structure:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Architecture Decisions:** [docs/adr/](docs/adr/)

---

## Success Criteria

You're ready to start development when:

```
✅ `python tests/validate_environment.py` passes 9/9 checks
✅ `make golden` completes without errors
✅ `make parity` shows 15/15 tests passing
✅ `benchmark/golden_stats.json` exists and is valid JSON
```

---

**Time invested:** ~30 minutes
**Confidence level:** 🟢 High (backed by regression tests)
**Next milestone:** Week 1 - Core refactoring with parity preservation

🎉 **Happy coding!**
