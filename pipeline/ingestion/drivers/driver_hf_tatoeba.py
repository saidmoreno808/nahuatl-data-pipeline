"""
Driver para Tatoeba MT dataset (Nahuatl-Spanish)
"""
from datasets import load_dataset
import logging

logger = logging.getLogger("driver_hf_tatoeba")

def ingest(source_url, **kwargs):
    """
    Ingesta Tatoeba MT dataset para pares nah-es.
    
    Args:
        source_url: 'Helsinki-NLP/tatoeba_mt'
    
    Returns:
        list: Lista de diccionarios con pares nah-es
    """
    logger.info(f"Cargando Tatoeba desde {source_url}...")
    
    try:
        # Tatoeba tiene pares de idiomas, buscar nah-es
        dataset = load_dataset(source_url, "nah-es")
        
        data = []
        
        for split in dataset.keys():
            for item in dataset[split]:
                translation = item.get('translation', {})
                data.append({
                    "nahuatl": translation.get('nah', ''),
                    "spanish": translation.get('es', ''),
                    "split": split
                })
        
        logger.info(f"✅ Tatoeba: {len(data)} pares cargados")
        return data
        
    except Exception as e:
        logger.warning(f"⚠️  Error con nah-es: {e}, intentando configuraciones alternativas...")
        
        # Intentar otras configuraciones
        configs = ['nah-spa', 'nah_Latn-spa_Latn', 'nah-es']
        
        for config in configs:
            try:
                dataset = load_dataset(source_url, config)
                data = []
                
                for split in dataset.keys():
                    for item in dataset[split]:
                        if 'translation' in item:
                            trans = item['translation']
                            # Detectar cuál es náhuatl
                            nah_text = trans.get('nah', trans.get('nah_Latn', ''))
                            es_text = trans.get('es', trans.get('spa', trans.get('spa_Latn', '')))
                            
                            if nah_text:
                                data.append({
                                    "nahuatl": nah_text,
                                    "spanish": es_text,
                                    "split": split
                                })
                
                if data:
                    logger.info(f"✅ Tatoeba ({config}): {len(data)} pares")
                    return data
                    
            except Exception as e2:
                continue
        
        logger.error(f"❌ No se pudo cargar Tatoeba con ninguna configuración")
        return []
