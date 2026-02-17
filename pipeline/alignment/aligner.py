from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Aligner:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        logger.info(f"Cargando modelo de alineamiento: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.sentences_es = []

    def build_index(self, sentences_es: list):
        """Construye el índice Faiss con oraciones en español."""
        self.sentences_es = sentences_es
        logger.info(f"Codificando {len(sentences_es)} oraciones en español...")
        embeddings = self.model.encode(sentences_es, convert_to_numpy=True, normalize_embeddings=True)
        
        # Faiss IndexFlatIP (Inner Product = Cosine Similarity si están normalizados)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        logger.info("Índice Faiss construido.")

    def align(self, sentences_nah: list, threshold=0.85):
        """Alinea oraciones en náhuatl contra el índice de español."""
        if not self.index:
            raise ValueError("Índice no construido. Llama a build_index primero.")
        
        logger.info(f"Alineando {len(sentences_nah)} oraciones en náhuatl...")
        embeddings_nah = self.model.encode(sentences_nah, convert_to_numpy=True, normalize_embeddings=True)
        
        # Buscar el vecino más cercano (k=1)
        D, I = self.index.search(embeddings_nah, 1)
        
        aligned_pairs = []
        for i, (score, idx) in enumerate(zip(D, I)):
            score = float(score[0])
            idx = int(idx[0])
            
            if score >= threshold:
                pair = {
                    "nah": sentences_nah[i],
                    "es": self.sentences_es[idx],
                    "score": score
                }
                aligned_pairs.append(pair)
                
        logger.info(f"Alineados {len(aligned_pairs)} pares (threshold={threshold})")
        return aligned_pairs
