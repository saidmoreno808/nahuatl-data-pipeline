import pandas as pd
from pathlib import Path
import logging

# Configuración
SOURCES_FILE = Path("sources.csv")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sources():
    """Carga el registro de fuentes."""
    if not SOURCES_FILE.exists():
        logger.error(f"No se encontró {SOURCES_FILE}")
        return pd.DataFrame()
    return pd.read_csv(SOURCES_FILE)

def get_pending_sources():
    """Obtiene las fuentes pendientes de ingesta."""
    df = load_sources()
    if df.empty:
        return []
    return df[df['ingestion_status'] == 'Pending'].to_dict('records')

def update_source_status(source_id, status):
    """Actualiza el estado de una fuente."""
    df = load_sources()
    if df.empty:
        return
    
    if source_id in df['source_id'].values:
        df.loc[df['source_id'] == source_id, 'ingestion_status'] = status
        df.to_csv(SOURCES_FILE, index=False)
        logger.info(f"Fuente {source_id} actualizada a {status}")
    else:
        logger.warning(f"Fuente {source_id} no encontrada")

if __name__ == "__main__":
    logger.info("=== Fase 1: Discovery ===")
    sources = load_sources()
    print(sources)
    logger.info(f"Total fuentes registradas: {len(sources)}")
    pending = get_pending_sources()
    logger.info(f"Fuentes pendientes: {len(pending)}")
