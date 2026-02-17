import json
import os
import random
import logging
import time
from typing import List, Dict
from gradio_client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Paths
SILVER_DATA_PATH = "data/silver/distilled_corpus_v1.jsonl" # Source for Gold Set
GOLD_SET_PATH = "data/gold/benchmark_test_set_es_nah.jsonl"
RIVAL_PREDICTIONS_PATH = "data/gold/rival_predictions_thermostatic_27b.jsonl"

# Rival Space
HF_SPACE_ID = "Thermostatic/neuraltranslate-27b-mt-nah-es"

def create_golden_set(source_path: str, output_path: str, sample_size: int = 500):
    """
    Creates a held-out test set from the silver data.
    """
    if os.path.exists(output_path):
        logger.info(f"Golden set already exists at {output_path}. Skipping creation.")
        return

    logger.info(f"Loading source data from {source_path}...")
    data = []
    if os.path.exists(source_path):
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # We only want items that have both 'es' and 'nah' for ground truth
                    if 'es' in item and 'nah' in item and len(item['es']) > 20: # Filter short noise
                        data.append(item)
                except:
                    continue
    
    if len(data) < sample_size:
        logger.warning(f"Not enough data to sample {sample_size}. Using all {len(data)} items.")
        sample = data
    else:
        # Shuffle and pick to ensure random distribution
        random.shuffle(data)
        sample = data[:sample_size]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Created Golden Set with {len(sample)} items at {output_path}")

def harvest_rival_predictions(input_path: str, output_path: str):
    """
    Sends Spanish sentences to the Rival HF Space and records predictions.
    """
    logger.info(f"Connecting to Rival Space: {HF_SPACE_ID}...")
    try:
        client = Client(HF_SPACE_ID)
    except Exception as e:
        logger.error(f"Failed to connect to Gradio Client: {e}")
        return

    # Load Golden Input
    with open(input_path, 'r', encoding='utf-8') as f:
        inputs = [json.loads(line) for line in f]
    
    logger.info(f"Harvesting predictions for {len(inputs)} items...")
    
    # Check existing to resume
    existing_preds = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_preds = sum(1 for _ in f)
    
    logger.info(f"Resuming from index {existing_preds}...")
    
    with open(output_path, 'a', encoding='utf-8') as f:
        for i, item in enumerate(inputs[existing_preds:], start=existing_preds):
            es_text = item.get('es', '')
            if not es_text: continue
            
            logger.info(f"[{i+1}/{len(inputs)}] Querying Rival: '{es_text[:30]}...'")
            
            try:
                # The API signature for translation spaces usually accepts text. 
                # We interpret the API from the Space's standard behavior.
                # If there are named endpoints, we might need to adjust (e.g. /predict).
                result = client.predict(
                    es_text,	# str  in 'Input' Textbox component
                    api_name="/predict" # Standard default, might need adjustment if space uses different name
                )
                
                # Result is usually the translated text
                prediction = result
                
                output_record = {
                    "original_es": es_text,
                    "ground_truth_nah": item.get('nah', ''),
                    "rival_prediction_nah": prediction,
                    "model_name": "neuraltranslate-27b"
                }
                
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                f.flush()
                
                # Be polite to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to predict item {i}: {e}")
                # Don't break, try next
                time.sleep(5)

if __name__ == "__main__":
    # 1. Create Data
    create_golden_set(SILVER_DATA_PATH, GOLD_SET_PATH)
    
    # 2. Attack!
    harvest_rival_predictions(GOLD_SET_PATH, RIVAL_PREDICTIONS_PATH)
