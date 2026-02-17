import fasttext
import logging
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_PATH = Path("models/lid.176.bin")

def download_model():
    """Descarga el modelo FastText si no existe."""
    if not MODEL_PATH.exists():
        logger.info("Descargando modelo LID FastText...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Modelo descargado.")

class LanguageIdentifier:
    def __init__(self):
        download_model()
        try:
            self.model = fasttext.load_model(str(MODEL_PATH))
        except Exception as e:
            logger.error(f"Error cargando modelo FastText: {e}")
            self.model = None

    def predict(self, text: str):
        """Predice el idioma y la confianza."""
        if not self.model:
            return "unknown", 0.0
        
        # FastText espera una sola línea sin saltos
        text = text.replace("\n", " ")
        try:
            predictions = self.model.predict(text)
            lang = predictions[0][0].replace("__label__", "")
            score = predictions[1][0]
            return lang, score
        except Exception as e:
            logger.error(f"Error en predicción LID: {e}")
            return "error", 0.0
