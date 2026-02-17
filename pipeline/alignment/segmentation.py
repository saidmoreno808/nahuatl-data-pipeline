import pysbd

class Segmenter:
    def __init__(self, lang="es"):
        self.seg = pysbd.Segmenter(language=lang, clean=False)

    def segment(self, text: str) -> list:
        """Segmenta texto en oraciones."""
        if not text:
            return []
        return self.seg.segment(text)
