"""
Driver para AmericasNLP 2021 dataset (Nahuatl)
"""
from datasets import load_dataset
import logging

logger = logging.getLogger("driver_hf_americasnlp")

def ingest(source_url, **kwargs):
    """
    Ingesta AmericasNLP 2021 dataset para Nahuatl.
    
    Args:
        source_url: 'AmericasNLP/americasnlp2021'
    
    Returns:
        list: Lista de diccionarios con datos de náhuatl
    """
    logger.info(f"Cargando AmericasNLP desde {source_url}...")
    
    try:
        # AmericasNLP tiene múltiples idiomas indígenas
        dataset = load_dataset(source_url, "nahuatl")
        
        data = []
        
        for split in dataset.keys():
            for item in dataset[split]:
                data.append({
                    "nahuatl": item.get('text', item.get('source', '')),
                    "spanish": item.get('target', ''),
                    "split": split,
                    "task": item.get('task', 'translation')
                })
        
        logger.info(f"✅ AmericasNLP: {len(data)} registros cargados")
        return data
        
    except Exception as e:
        logger.warning(f"⚠️  Error con 'nahuatl': {e}, intentando alternativas...")
        
        # Intentar configuraciones alternativas
        configs = ['nah', 'nahuatl', 'all']
        
        for config in configs:
            try:
                dataset = load_dataset(source_url, config)
                data = []
                
                for split in dataset.keys():
                    for item in dataset[split]:
                        # Extraer texto de náhuatl
                        text = item.get('text', item.get('source', item.get('sentence', '')))
                        
                        if text:
                            data.append({
                                "text": text,
                                "metadata": item,
                                "split": split
                            })
                
                if data:
                    logger.info(f"✅ AmericasNLP ({config}): {len(data)} registros")
                    return data
                    
            except Exception as e2:
                continue
        
        logger.error(f"❌ No se pudo cargar AmericasNLP")
        return []
