"""
Driver para corpus bíblico en Náhuatl
"""
import requests
import logging
import re

logger = logging.getLogger("driver_bible")

def ingest(source_url, **kwargs):
    """
    Ingesta corpus bíblico en Náhuatl.
    
    Args:
        source_url: 'bible-nahuatl-central'
    
    Returns:
        list: Lista de diccionarios con versículos
    """
    logger.info(f"Cargando corpus bíblico...")
    
    # URLs de Biblias en Náhuatl disponibles públicamente
    bible_sources = [
        "https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/nah-x-bible.txt",
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Nahuatl.xml",
    ]
    
    data = []
    
    for url in bible_sources:
        try:
            logger.info(f"Intentando descargar de {url}...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                content = response.text
                
                # Parsear según formato
                if url.endswith('.txt'):
                    # Formato texto plano, separar por líneas
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line and len(line) > 10:  # Filtrar líneas vacías/cortas
                            data.append({
                                "text": line,
                                "source": "bible",
                                "verse_id": i
                            })
                
                elif url.endswith('.xml'):
                    # Parsear XML (simplificado)
                    verses = re.findall(r'<verse[^>]*>(.*?)</verse>', content, re.DOTALL)
                    for i, verse in enumerate(verses):
                        # Limpiar tags HTML
                        clean_verse = re.sub(r'<[^>]+>', '', verse).strip()
                        if clean_verse and len(clean_verse) > 10:
                            data.append({
                                "text": clean_verse,
                                "source": "bible",
                                "verse_id": i
                            })
                
                logger.info(f"✅ Descargados {len(data)} versículos de {url}")
                break  # Si uno funciona, no intentar otros
                
        except Exception as e:
            logger.warning(f"⚠️  Error con {url}: {e}")
            continue
    
    if not data:
        logger.warning("⚠️  No se pudo descargar corpus bíblico, usando datos de ejemplo...")
        # Datos de ejemplo si no se puede descargar
        data = [
            {"text": "In ipan pehuallotl, Dios quichiuh ilhuicatl ihuan tlaltipactli.", "source": "bible_example", "verse_id": 1},
            {"text": "Ihuan Dios quitoh: Machihua tlanextli.", "source": "bible_example", "verse_id": 2},
        ]
    
    logger.info(f"✅ Total corpus bíblico: {len(data)} registros")
    return data
