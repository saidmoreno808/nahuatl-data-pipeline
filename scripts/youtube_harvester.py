import os
import time
import json
import logging
import argparse
import glob
import subprocess
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("harvester.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YoutubeHarvester:
    def __init__(self, output_dir: str = "data/bronze/audio_temp"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parse Keys: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3...
        self.api_keys = []
        
        # Primary Key
        primary_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if primary_key:
            self.api_keys.append(primary_key)
        
        # Secondary Keys (Environment only)
        for i in range(2, 10): # Check up to GEMINI_API_KEY_9
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
        
        if not self.api_keys:
            raise ValueError("API Key not found. Set GEMINI_API_KEY env vars.")
        
        self.current_key_index = 0
        self.model_name = "gemini-2.5-flash"
        self._configure_current_key()
        
    def _configure_current_key(self):
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Switched to API Key #{self.current_key_index + 1}")

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()

    def search_and_download_audio(self, query: str, limit: int = 1, date_after: str = "20240101"):
        """
        Uses yt-dlp to download audio from videos matching the query OR from a direct URL (Video/Channel/Playlist).
        """
        logger.info(f"Targeting: '{query}' (Limit: {limit})...")
        
        import sys
        
        target = query
        is_url = query.startswith("http")
        
        if not is_url:
            target = f"ytsearch{limit}:{query}"
            logger.info(f"Mode: SEARCH (Date > {date_after})")
        else:
             logger.info("Mode: DIRECT URL (Video/Channel/Playlist)")

        command = [
            sys.executable, "-m", "yt_dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "--output", f"{self.output_dir}/%(id)s.%(ext)s",
            "--max-downloads", str(limit),
            # "--dateafter", date_after, # Only apply date filter to search? Or keep for channels too? keeping off for now based on previous edits
            "--print-to-file", "%(webpage_url)s | %(channel)s | %(title)s", "data/silver/harvested_sources_log.txt",
            "--ignore-errors", # Skip video if download fails
            target
        ]
        
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True) # check=False to continue if one video fails
            logger.info("yt-dlp output:\n" + result.stdout)
            if result.stderr:
                logger.warning("yt-dlp stderr:\n" + result.stderr)
        except Exception as e:
            logger.error(f"yt-dlp execution error: {e}")

    def transcribe_and_translate(self, audio_path: str) -> Dict:
        """
        Uploads audio to Gemini and requests SOTA transcription/translation.
        Retries with key rotation if quota is exceeded.
        """
        max_attempts = len(self.api_keys) * 2 # Try each key twice efficiently
        
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt+1}/{max_attempts} for {os.path.basename(audio_path)} using Key #{self.current_key_index + 1}")
            
            try:
                # 1. Upload file (Must re-upload per key if keys are from different projects)
                logger.info(f"Uploading {audio_path}...")
                audio_file = genai.upload_file(audio_path)
                
                # Wait for processing
                while audio_file.state.name == "PROCESSING":
                    time.sleep(2)
                    audio_file = genai.get_file(audio_file.name)
                
                if audio_file.state.name == "FAILED":
                    logger.error("Audio processing failed on server side.")
                    # If failed on server, maybe rotating won't help, but we can try once more
                    self._rotate_key()
                    continue

                logger.info("Audio processed. Generating transcript...")
                
                prompt = """
                Listen to this audio carefully. It may contain Indigenous languages (Nahuatl or Maya) and Spanish.
                
                Your task is to create a structured dataset from this audio.
                Extract meaningful sentences or phrases.
                
                For each phrase, output a JSON object:
                {
                    "original_audio_text": "The exact text heard (Nahuatl/Maya/Spanish)",
                    "detected_language": "nah" or "myn" or "es",
                    "es_translation": "The Spanish translation (if original was Indigenous) or the same text (if original was Spanish)",
                    "nah_translation": "The Nahuatl translation (generate if missing)"
                }
                
                Output strictly a valid JSON list.
                """
                
                response = self.model.generate_content(
                    [audio_file, prompt],
                    generation_config={"response_mime_type": "application/json"}
                )
                
                # Cleanup remote file
                try:
                    genai.delete_file(audio_file.name)
                except: pass

                return json.loads(response.text)

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "resource" in error_str:
                    logger.warning(f"Key #{self.current_key_index + 1} exhausted ({e}). Rotating...")
                    self._rotate_key()
                    # Retry loop continues immediately with new key
                else:
                    logger.error(f"Gemini processing failed for {audio_path}: {e}")
                    # If it's not a quota error, it might be a bad file or other issue. 
                    # We usually shouldn't rotate for Logic errors, but let's try one rotation just in case.
                    if attempt < 2:
                        self._rotate_key()
                    else:
                        return None
        
        return None

    def run_pipeline(self, query: str, limit: int, output_file: str):
        # 1. Download
        self.search_and_download_audio(query, limit)
        
        # 2. Process Files
        mp3_files = glob.glob(os.path.join(self.output_dir, "*.mp3"))
        logger.info(f"Found {len(mp3_files)} audio files to process.")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for audio_path in mp3_files:
                data = self.transcribe_and_translate(audio_path)
                if data:
                    if isinstance(data, list):
                        for item in data:
                            item['source_file'] = os.path.basename(audio_path)
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                
                # Cleanup local file after processing to save space
                try:
                    os.remove(audio_path)
                    logger.info(f"Deleted local temp file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {audio_path}: {e}")
    def run_pipeline(self, query: str, limit: int, output_file: str):
        # 1. Download
        self.search_and_download_audio(query, limit)
        
        # 2. Process Files
        mp3_files = glob.glob(os.path.join(self.output_dir, "*.mp3"))
        logger.info(f"Found {len(mp3_files)} audio files to process.")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for audio_path in mp3_files:
                data = self.transcribe_and_translate(audio_path)
                if data:
                    if isinstance(data, list):
                        for item in data:
                            item['source_file'] = os.path.basename(audio_path)
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                
                # Cleanup local file after processing to save space
                try:
                    os.remove(audio_path)
                    logger.info(f"Deleted local temp file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {audio_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="YouTube Harvester (Audio -> Gemini)")
    parser.add_argument("--query", required=True, help="Search query (e.g. 'clases náhuatl')")
    parser.add_argument("--limit", type=int, default=3, help="Max videos to download")
    parser.add_argument("--output", default="data/silver/youtube_harvest.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    harvester = YoutubeHarvester()
    harvester.run_pipeline(args.query, args.limit, args.output)

if __name__ == "__main__":
    main()
