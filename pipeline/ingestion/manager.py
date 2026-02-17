import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging
# Configuración
BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class IngestionManager:
    """Gestiona el almacenamiento en la capa BRONZE."""
    
    def __init__(self):
        self.base_dir = BRONZE_DIR
    def generate_data_id(self, source_url: str) -> str:
        """Genera un ID único basado en la URL y timestamp."""
        timestamp = datetime.utcnow().isoformat()
        raw_string = f"{source_url}_{timestamp}"
        return hashlib.sha256(raw_string.encode()).hexdigest()[:16]
    def save_raw_data(self, data: bytes, source_meta: dict, extension: str = "bin") -> str:
        """Guarda datos crudos y metadatos en Bronze."""
        data_id = self.generate_data_id(source_meta['source_url'])
        
        # Rutas
        raw_path = self.base_dir / f"{data_id}.{extension}"
        meta_path = self.base_dir / f"{data_id}.meta.json"
        
        # Guardar datos crudos
        with open(raw_path, "wb") as f:
            f.write(data)
            
        # Preparar metadatos completos
        meta = {
            "data_id": data_id,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
            "file_path": str(raw_path),
            **source_meta
        }
        
        # Guardar metadatos
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        logger.info(f"[BRONZE] Guardado {data_id} ({extension})")
        return data_id
    def save_json_data(self, data: list, source_meta: dict) -> str:
        """Helper para guardar datos JSON como raw bytes."""
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        json_bytes = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False).encode('utf-8')
        return self.save_raw_data(json_bytes, source_meta, extension="json")