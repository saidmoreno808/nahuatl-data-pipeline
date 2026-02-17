import unicodedata
import re

def normalize_unicode(text: str) -> str:
    """Aplica normalización Unicode NFC."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def clean_regex(text: str) -> str:
    """Limpieza básica con Regex."""
    if not isinstance(text, str):
        return ""
    
    # Colapsar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)
    
    # Eliminar artefactos comunes (ej. "Página 5")
    text = re.sub(r'Página \d+', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def process_text(text: str) -> str:
    """Pipeline de limpieza estándar."""
    text = normalize_unicode(text)
    text = clean_regex(text)
    return text
