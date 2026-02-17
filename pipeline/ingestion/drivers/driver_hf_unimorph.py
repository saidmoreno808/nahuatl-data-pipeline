"""
Driver para UniMorph Nahuatl (morfología)
"""
from datasets import load_dataset
import logging

logger = logging.getLogger("driver_hf_unimorph")

def ingest(source_url, **kwargs):
    """
    Ingesta UniMorph dataset para Nahuatl (datos morfológicos).
    
    Args:
        source_url: 'unimorph/nahuatl'
    
    Returns:
        list: Lista de diccionarios con formas morfológicas
    """
    logger.info(f"Cargando UniMorph desde {source_url}...")
    
    try:
        dataset = load_dataset(source_url)
        
        data = []
        
        for split in dataset.keys():
            for item in dataset[split]:
                # UniMorph tiene lemma, forma inflectada, y tags morfológicos
                data.append({
                    "lemma": item.get('lemma', ''),
                    "form": item.get('form', ''),
                    "tags": item.get('tags', ''),
                    "split": split
                })
        
        logger.info(f"✅ UniMorph: {len(data)} formas morfológicas")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error cargando UniMorph: {e}")
        return []
