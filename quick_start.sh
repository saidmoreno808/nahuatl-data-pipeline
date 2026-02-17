#!/bin/bash
# Quick Start Script para Orquestación Híbrida

echo "========================================"
echo "QUICK START: Orquestación Híbrida Gemini"
echo "========================================"

# 1. Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python no encontrado"
    exit 1
fi

echo "[1/5] Python OK"

# 2. Instalar dependencias
echo "[2/5] Instalando dependencias..."
pip install -q google-generativeai kaggle tqdm

# 3. Verificar API Key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "[3/5] GEMINI_API_KEY no configurada"
    read -p "Ingresa tu Gemini API Key: " GEMINI_API_KEY
    export GEMINI_API_KEY
else
    echo "[3/5] GEMINI_API_KEY configurada"
fi

# 4. Verificar Kaggle credentials
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "[4/5] Kaggle credentials no encontradas"
    echo "Descarga kaggle.json de: https://www.kaggle.com/settings"
    read -p "Path a kaggle.json: " KAGGLE_PATH
    mkdir -p "$HOME/.kaggle"
    cp "$KAGGLE_PATH" "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
else
    echo "[4/5] Kaggle credentials OK"
fi

# 5. Ejecutar orquestación
echo "[5/5] Iniciando orquestación..."
echo ""

read -p "Kaggle dataset (usuario/dataset-name): " DATASET
read -p "Batch size [15000]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-15000}

read -p "Budget diario [$10.00]: " BUDGET
BUDGET=${BUDGET:-10.0}

python orchestrate_hybrid_dpo.py \
    --kaggle-dataset "$DATASET" \
    --batch-size "$BATCH_SIZE" \
    --budget "$BUDGET"

echo ""
echo "========================================"
echo "Orquestación completada!"
echo "Resultados en: ./dpo_output_hybrid/"
echo "========================================"
