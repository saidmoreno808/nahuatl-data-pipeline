import os
import json
import glob
import pandas as pd
import logging

from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SILVER_DIR = "data/silver"
DIAMOND_DIR = "data/diamond"
GOLD_DIR = "data/gold"

OUTPUT_TRAIN = os.path.join(GOLD_DIR, "train_v1.jsonl")
OUTPUT_VAL = os.path.join(GOLD_DIR, "validation_v1.jsonl")
OUTPUT_TEST = os.path.join(GOLD_DIR, "test_v1.jsonl")

def normalize_record(record: Dict) -> Dict:
    """
    Standardizes a record to have 'es', 'nah', or 'myn' keys.
    Handles variations from different Harvester/Distiller versions.
    """
    normalized = {}
    
    # Extract Spanish (or DPO Prompt)
    if 'es' in record:
        normalized['es'] = record['es']
    elif 'es_translation' in record:
        normalized['es'] = record['es_translation']
    elif 'original_es' in record:
        normalized['es'] = record['original_es']
    elif 'prompt' in record: # DPO format
        normalized['es'] = record['prompt']
    
    # Extract Nahuatl (or DPO Chosen)
    if 'nah' in record:
        normalized['nah'] = record['nah']
    elif 'nah_translation' in record:
        normalized['nah'] = record['nah_translation']
    elif 'original_audio_text' in record and record.get('detected_language') == 'nah':
        normalized['nah'] = record['original_audio_text']
    elif 'chosen' in record: # DPO format
        normalized['nah'] = record['chosen']
        
    # Extract Maya
    if 'myn' in record:
        normalized['myn'] = record['myn']
    elif 'myn_translation' in record:
        normalized['myn'] = record['myn_translation']
    elif 'original_audio_text' in record and record.get('detected_language') == 'myn':
        normalized['myn'] = record['original_audio_text']
        
    # Add metadata if available
    if 'source_file' in record:
        normalized['source'] = record['source_file']
    if 'category' in record:
        normalized['category'] = record['category']
        
    return normalized

def load_and_merge_data() -> pd.DataFrame:
    all_records = []
    
    # 1. Load Silver (Distilled + Harvested + Raw Dumps)
    silver_files = glob.glob(os.path.join(SILVER_DIR, "*.jsonl"))
    silver_json_files = glob.glob(os.path.join(SILVER_DIR, "*.json")) # Catch legacy dumps
    
    logger.info(f"Found {len(silver_files)} JSONL and {len(silver_json_files)} JSON datasets in Silver layer.")
    
    # Process JSONL files
    for fpath in silver_files:
        filename = os.path.basename(fpath)
        logger.info(f"Processing JSONL {filename}...")
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        norm = normalize_record(record)
                        norm['layer'] = 'silver'
                        norm['origin_file'] = filename
                        if norm.get('es') and (norm.get('nah') or norm.get('myn')):
                            all_records.append(norm)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    # Process JSON dumps (Py-Elotl / Axolotl formats)
    for fpath in silver_json_files:
        filename = os.path.basename(fpath)
        logger.info(f"Processing JSON dump {filename}...")
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check for 'items' list (Standard Bronze->Silver format)
                items = data.get('items', [])
                if not items and isinstance(data, list):
                    items = data # Handle if root is just a list
                    
                for item in items:
                    # Extraction Logic for this schema
                    # Usually nested in 'original' or direct keys
                    raw_rec = {}
                    
                    # Try 'original' key (Py-Elotl format)
                    if 'original' in item:
                        orig = item['original']
                        raw_rec['es'] = orig.get('es', orig.get('sp')) # Handle 'sp' key
                        raw_rec['nah'] = orig.get('nah')
                        raw_rec['myn'] = orig.get('myn')
                    else:
                        # Direct keys
                        raw_rec = item
                        
                    norm = normalize_record(raw_rec)
                    norm['layer'] = 'silver_dump'
                    norm['origin_file'] = filename
                    
                    if norm.get('es') and (norm.get('nah') or norm.get('myn')):
                        all_records.append(norm)
                        
        except Exception as e:
            logger.error(f"Error reading JSON dump {filename}: {e}")

    # 2. Load Diamond (High Quality / Manual / Synthetic)
    diamond_files = glob.glob(os.path.join(DIAMOND_DIR, "*.jsonl"))
    logger.info(f"Found {len(diamond_files)} datasets in Diamond layer.")
    
    for fpath in diamond_files:
        filename = os.path.basename(fpath)
        logger.info(f"Processing {filename}...")
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        norm = normalize_record(record)
                        norm['layer'] = 'diamond' # Higher relevance
                        norm['origin_file'] = filename
                        
                        if norm.get('es') and (norm.get('nah') or norm.get('myn')):
                            all_records.append(norm)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    df = pd.DataFrame(all_records)
    logger.info(f"Total raw records loaded: {len(df)}")
    return df

def clean_and_split(df: pd.DataFrame):
    # Deduplicate based on Spanish + Nahuatl + Maya combination
    # We prioritize Diamond layer duplicates over Silver if conflict (keep last or sort)
    
    # Sort so Diamond is at the bottom (keep='last' will keep Diamond)
    df['layer_rank'] = df['layer'].map({'silver': 0, 'diamond': 1})
    df = df.sort_values('layer_rank')
    
    initial_count = len(df)
    
    # Create a unique key for deduplication
    df['dedup_key'] = df['es'].str.lower().str.strip() + "_" + \
                      df['nah'].fillna('').str.lower().str.strip() + "_" + \
                      df['myn'].fillna('').str.lower().str.strip()
                      
    df = df.drop_duplicates(subset='dedup_key', keep='last')
    
    logger.info(f"Deduplication removed {initial_count - len(df)} records. Final count: {len(df)}")
    
    # Stats
    nah_count = df['nah'].notna().sum()
    myn_count = df['myn'].notna().sum()
    logger.info(f"Language Distribution: Nahuatl={nah_count}, Maya={myn_count}")
    
    # Verify we actually have data
    if len(df) == 0:
        logger.error("No data found! Aborting.")
        return

    # Split
    # Manual random split instead of sklearn to avoid dependency issues
    import random
    random.seed(42)
    
    # Shuffle indices
    indices = df.index.tolist()
    random.shuffle(indices)
    
    total = len(df)
    train_end = int(total * 0.9)
    val_end = int(total * 0.95)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train = df.loc[train_indices]
    val = df.loc[val_indices]
    test = df.loc[test_indices]
    
    logger.info(f"Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # Save
    os.makedirs(GOLD_DIR, exist_ok=True)
    
    def save_jsonl(dataframe, path):
        # Clean up internal columns
        export_df = dataframe.drop(columns=['layer_rank', 'dedup_key'])
        # Convert to records
        records = export_df.to_dict(orient='records')
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                # Remove None values to make clean JSON
                clean_r = {k: v for k, v in r.items() if pd.notna(v)}
                f.write(json.dumps(clean_r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {path}")

    save_jsonl(train, OUTPUT_TRAIN)
    save_jsonl(val, OUTPUT_VAL)
    save_jsonl(test, OUTPUT_TEST)
    
    logger.info("Dataset Unification Complete. Ready for Kaggle.")

if __name__ == "__main__":
    df = load_and_merge_data()
    clean_and_split(df)
