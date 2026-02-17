"""
Driver para FLORES-200 dataset (Nahuatl)
"""
from datasets import load_dataset
import logging

logger = logging.getLogger("driver_hf_flores")

def ingest(source_url, **kwargs):
    """
    Ingesta FLORES-200 dataset para Nahuatl.
    
    Args:
        source_url: 'facebook/flores'
    
    Returns:
        list: Lista de diccionarios con pares nah-es
    """
    logger.info(f"Cargando FLORES-200 desde {source_url}...")
    
    try:
        # FLORES-200 tiene múltiples idiomas, filtrar náhuatl
        dataset = load_dataset(source_url, "nah_Latn")
        
        data = []
        
        # FLORES tiene splits: dev, devtest
        for split in ['dev', 'devtest']:
            if split in dataset:
                for item in dataset[split]:
                    data.append({
                        "nahuatl": item.get('sentence', ''),
                        "spanish": item.get('sentence_es', ''),  # Si existe traducción
                        "split": split,
                        "id": item.get('id', '')
                    })
        
        logger.info(f"✅ FLORES-200: {len(data)} registros cargados")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error cargando FLORES-200: {e}")
        # Intentar método alternativo
        try:
            dataset = load_dataset(source_url, "all")
            data = []
            
            for split in dataset.keys():
                for item in dataset[split]:
                    # Buscar campos de náhuatl
                    if 'nah' in str(item).lower():
                        data.append({
                            "text": str(item),
                            "split": split
                        })
            
            logger.info(f"✅ FLORES-200 (alternativo): {len(data)} registros")
            return data
            
        except Exception as e2:
            logger.error(f"❌ Error en método alternativo: {e2}")
            return []
