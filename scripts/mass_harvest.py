import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUERIES = [
    # --- NAHUATL ---
    ("https://www.youtube.com/@Kerneup", 5),
    ("https://www.youtube.com/@SECVERoficial", 5),
    ("https://www.youtube.com/@impepac.morelos", 5),
    ("https://www.youtube.com/watch?v=90cpcQ3_4D0", 1),
    
    # --- MAYA ---
    ("https://www.youtube.com/watch?v=X2Dj1cjaue0", 1),
    ("https://www.youtube.com/channel/UCo6obsptMAcu89SdlOrIlqQ", 5),
    ("https://www.youtube.com/watch?v=U8q7mXGHGH0", 1),
    ("https://www.youtube.com/@ConversationalMaya", 5),
    ("https://www.youtube.com/watch?v=uCZhtut6enE", 1),
    ("https://www.youtube.com/watch?v=ArfQLN9Chd0", 1),
    ("https://www.youtube.com/playlist?list=PLhAajGvZOKgA_nO6oXCJwG26TgdqbueX_", 10),
    ("https://www.youtube.com/playlist?list=PL84PsbyKaUywGGLiZPPxFYfPYJq9_fFIc", 10),
    ("https://www.youtube.com/watch?v=cUNLmxQc9Xg", 1),
    ("https://www.youtube.com/@RepublicaMaya", 5),

    # --- Previous Queries (Backlog) ---
    ("aprender maya yucateco", 5),
    ("cuentos en lengua indígena méxico", 5),
    ("noticias en náhuatl", 5),
    ("poesía náhuatl audio", 3)
]

def run_harvest():
    total_videos = 0
    for query, limit in QUERIES:
        logger.info(f"--- Harvesting: '{query}' (Limit: {limit}) ---")
        try:
            # Call the harvester script properly
            subprocess.run([
                "python", "scripts/youtube_harvester.py",
                "--query", query,
                "--limit", str(limit),
                "--output", "data/silver/youtube_harvest_corpus.jsonl"
            ], check=True)
            total_videos += limit
            
            # Pause to be polite to API and YouTube
            logger.info("Sleeping 10s...")
            time.sleep(10)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to harvest '{query}': {e}")
            
    logger.info(f"Mass harvest complete. Target videos: ~{total_videos}")

if __name__ == "__main__":
    run_harvest()
