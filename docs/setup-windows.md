# Windows Development Setup Guide

## Critical Prerequisites

**⚠️ ALL development MUST happen in WSL2, NOT Windows native**

### Why WSL2 is Mandatory

| Issue | Windows Native | WSL2 |
|-------|---------------|------|
| Line endings | CRLF (breaks scripts) | LF (Unix-standard) |
| Path resolution | `C:\Users\...` | `/mnt/c/users/...` |
| Docker volumes | Slow, permission issues | Native performance |
| Unicode handling | Code page issues | UTF-8 by default |
| Shell scripts | Require Git Bash/WSL | Native execution |

---

## 1. Install WSL2 Ubuntu 22.04

### PowerShell (Administrator)
```powershell
# Enable WSL
wsl --install -d Ubuntu-22.04

# Verify installation
wsl --list --verbose
# Should show: Ubuntu-22.04  Running  2

# Set as default
wsl --set-default Ubuntu-22.04
```

### First Boot Configuration
```bash
# Inside WSL2 terminal
# Create user (prompted automatically)
# Then update system
sudo apt update && sudo apt upgrade -y
```

---

## 2. Install Docker Desktop

1. Download from https://www.docker.com/products/docker-desktop
2. Enable WSL2 backend during installation
3. **Settings → Resources → WSL Integration**
   - Enable integration with Ubuntu-22.04
4. Verify from WSL2:
   ```bash
   docker --version
   docker compose version
   ```

---

## 3. Configure Git in WSL2

```bash
# Core settings
git config --global user.name "Said Moreno"
git config --global user.email "your-email@example.com"

# CRITICAL: Force LF line endings
git config --global core.autocrlf input
git config --global core.eol lf

# Better diff for Python
git config --global diff.python.xfuncname "^[ \t]*((class|def)[ \t].*)$"

# Verify
git config --list | grep -E "autocrlf|eol"
```

### Clone Repository INSIDE WSL2

```bash
# ❌ WRONG (Windows filesystem)
cd /mnt/c/Users/YourName/Documents/
git clone https://github.com/youruser/corc-nah.git

# ✅ CORRECT (WSL2 native filesystem)
cd ~
mkdir -p projects
cd projects
git clone https://github.com/youruser/corc-nah.git
cd corc-nah

# Verify line endings were applied
file scripts/unify_datasets.py
# Should output: ASCII text, with LF line terminators
```

---

## 4. Install Python 3.10

```bash
# Add deadsnakes PPA for Python 3.10
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.10 + dev headers
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Verify
python3.10 --version  # Should show: Python 3.10.x
```

---

## 5. Install Project Dependencies

### Create Virtual Environment
```bash
cd ~/projects/corc-nah

# Create venv
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Requirements
```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Or use Poetry (recommended)
pip install poetry
poetry install
```

### Verify Installation
```bash
# Test critical imports
python -c "import pandas; print(pandas.__version__)"
python -c "import yt_dlp; print(yt_dlp.version.__version__)"
python -c "from youtube_transcript_api import YouTubeTranscriptApi; print('OK')"
```

---

## 6. Configure VS Code

### Install Extensions
- **WSL** (ms-vscode-remote.remote-wsl)
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Black Formatter** (ms-python.black-formatter)
- **EditorConfig** (editorconfig.editorconfig)

### Open Project in WSL
```bash
# From WSL terminal
code ~/projects/corc-nah
```

### Workspace Settings (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "files.eol": "\n",
    "files.encoding": "utf8",
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

---

## 7. Initialize Pre-Commit Hooks (Optional)

```bash
# Install pre-commit framework
pip install pre-commit

# Install hooks
pre-commit install

# Test
pre-commit run --all-files
```

---

## 8. Verify Full Setup

### Run Setup Validation Script
```bash
cd ~/projects/corc-nah
python tests/validate_environment.py
```

Expected output:
```
✓ Python 3.10.x detected
✓ Virtual environment active
✓ All required packages installed
✓ Git configured with LF line endings
✓ UTF-8 encoding enabled
✓ Docker accessible
✓ WSL2 native filesystem detected
```

---

## 9. Day 0: Generate Golden Dataset

```bash
# Navigate to project root
cd ~/projects/corc-nah

# Activate venv
source .venv/bin/activate

# Run legacy pipeline with deterministic seed
python scripts/unify_datasets.py --seed 42

# Copy outputs to benchmark/
mkdir -p benchmark/
cp data/gold/train_v1.jsonl benchmark/golden_train_v1.jsonl
cp data/gold/validation_v1.jsonl benchmark/golden_validation_v1.jsonl
cp data/gold/test_v1.jsonl benchmark/golden_test_v1.jsonl

# Generate checksums
md5sum benchmark/golden_*.jsonl > benchmark/checksums.txt

# Generate statistics
python benchmark/generate_stats.py

# Run parity tests (should pass 100%)
pytest tests/integration/test_parity_with_legacy.py -v
```

---

## Common Issues & Solutions

### Issue: Scripts fail with `\r\n` errors
**Cause:** Files cloned in Windows before WSL2 setup
**Solution:**
```bash
# Re-clone in WSL2 native filesystem
rm -rf ~/projects/corc-nah
cd ~/projects
git clone https://github.com/youruser/corc-nah.git

# Or fix existing files
find . -type f -name "*.py" -o -name "*.sh" | xargs dos2unix
```

### Issue: Docker can't mount volumes
**Cause:** Using Windows path in WSL2
**Solution:**
```bash
# ❌ WRONG
docker run -v /mnt/c/Users/Name/data:/data image

# ✅ CORRECT
docker run -v ~/projects/corc-nah/data:/data image
```

### Issue: UTF-8 decoding errors
**Cause:** Windows code page encoding
**Solution:**
```bash
# Set locale in WSL2
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Add to ~/.bashrc
echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
```

### Issue: Permission denied on scripts
**Solution:**
```bash
chmod +x scripts/*.sh
git update-index --chmod=+x scripts/*.sh
```

---

## Performance Tips

1. **Store code in WSL2 filesystem** (`~/projects/`), not `/mnt/c/`
   - 5-10x faster file I/O
   - Native symlink support
   - Proper Unix permissions

2. **Use VS Code Remote - WSL**
   - Better IntelliSense performance
   - Native terminal integration

3. **Increase WSL2 memory** (optional)
   ```powershell
   # Create C:\Users\YourName\.wslconfig
   [wsl2]
   memory=8GB
   processors=4
   ```

---

## Next Steps

1. Read [docs/architecture.md](./architecture.md)
2. Review [docs/adr/001-why-sqlite.md](./adr/001-why-sqlite.md)
3. Run `make test` to verify setup
4. Start refactoring with Tier 1 tasks
