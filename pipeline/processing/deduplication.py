from datasketch import MinHash, MinHashLSH
import re

class Deduplicator:
    def __init__(self, threshold=0.9, num_perm=128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.seen_hashes = set()

    def _get_minhash(self, text: str) -> MinHash:
        """Calcula MinHash para un texto."""
        m = MinHash(num_perm=self.num_perm)
        # Tokenización simple por espacios para n-gramas (o palabras)
        tokens = set(text.split())
        for token in tokens:
            m.update(token.encode('utf8'))
        return m

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """Verifica si el texto es duplicado y lo indexa si es nuevo."""
        m = self._get_minhash(text)
        
        # Consultar LSH
        result = self.lsh.query(m)
        
        if result:
            return True # Es duplicado de algo ya visto
        
        # Si no es duplicado, insertar
        self.lsh.insert(doc_id, m)
        return False
