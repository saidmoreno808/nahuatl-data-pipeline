import logging
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import pdfplumber
import json
from .manager import IngestionManager

logger = logging.getLogger(__name__)
manager = IngestionManager()

def driver_hf_dataset(source_row):
    """Ingesta datasets de Hugging Face."""
    repo_id = source_row['source_url']
    logger.info(f"Ingestando HF: {repo_id}")
    
    try:
        # Intentar cargar train
        try:
            ds = load_dataset(repo_id, split='train')
        except:
            ds = load_dataset(repo_id, split='train_sft') # Caso ShareGPT
            
        # Convertir a lista de dicts para guardar como JSON crudo
        data = ds.to_pandas().to_dict(orient='records')
        
        manager.save_json_data(data, source_row)
        return True
    except Exception as e:
        logger.error(f"Error en driver_hf_dataset: {e}")
        return False

def driver_pdf_plumber(source_row):
    """Ingesta PDF local (copia el archivo a Bronze)."""
    pdf_path = Path(source_row['source_url'])
    logger.info(f"Ingestando PDF: {pdf_path}")
    
    if not pdf_path.exists():
        logger.error(f"PDF no encontrado: {pdf_path}")
        return False
        
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        manager.save_raw_data(pdf_bytes, source_row, extension="pdf")
        return True
    except Exception as e:
        logger.error(f"Error en driver_pdf_plumber: {e}")
        return False

def driver_pyelotl(source_row):
    """Ingesta datos de Py-Elotl."""
    logger.info("Ingestando Py-Elotl...")
    try:
        import elotl.corpus
        corpus = elotl.corpus.load('axolotl')
        
        # Py-Elotl devuelve lista de listas/tuplas. Estandarizar a dict.
        data = [{"es": item[0], "nah": item[1]} for item in corpus]
        
        manager.save_json_data(data, source_row)
        return True
    except ImportError:
        logger.error("py-elotl no instalado")
        return False
    except Exception as e:
        logger.error(f"Error en driver_pyelotl: {e}")
        return False

# Mapa de drivers
DRIVERS = {
    'driver_hf_dataset': driver_hf_dataset,
    'driver_pdf_plumber': driver_pdf_plumber,
    'driver_pyelotl': driver_pyelotl
}

def run_ingestion(source_row):
    driver_name = source_row['ingestion_driver']
    driver = DRIVERS.get(driver_name)
    
    if driver:
        return driver(source_row)
    else:
        logger.error(f"Driver no encontrado: {driver_name}")
        return False
