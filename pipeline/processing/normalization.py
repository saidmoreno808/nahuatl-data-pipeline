import re

class Normalizer:
    """
    Normalizador ortográfico para Náhuatl.
    NOTA: Implementación basada en Regex como fallback a FSTs (Pynini)
    debido a limitaciones de plataforma en Windows.
    """
    
    def __init__(self, variant="central"):
        self.variant = variant

    def normalize(self, text: str) -> str:
        if self.variant == "central" or self.variant == "clasico":
            return self._normalize_central(text)
        elif self.variant == "huasteca":
            return self._normalize_huasteca(text)
        return text

    def _normalize_central(self, text: str) -> str:
        # Reglas básicas de estandarización para Clásico/Central
        # k/c/qu -> k (fonético simplificado para ML, opcional)
        # O mantener ortografía clásica: c ante a/o/u, qu ante e/i
        
        # Ejemplo: Estandarizar saltillo 'h' vs apostrofe
        text = text.replace("'", "h")
        text = text.replace("’", "h")
        
        # Estandarizar w -> hu
        text = re.sub(r'\bw', 'hu', text, flags=re.IGNORECASE)
        
        return text

    def _normalize_huasteca(self, text: str) -> str:
        # Huasteca usa 'w' frecuentemente
        text = text.replace("hu", "w")
        return text
