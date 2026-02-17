import os
import json
import time
import asyncio
import logging
import argparse
import re
from typing import List, Dict, Optional, Union
import io

import google.generativeai as genai
from google.api_core import exceptions
import httpx
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentExtractor:
    @staticmethod
    def clean_text(text: str) -> str:
        """normalize whitespace."""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks suitable for distillation."""
        # Simple splitting by sentences or approx length
        # For now, just split by period + space to keep it simple and keep sentences intact-ish
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += sent + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    @staticmethod
    async def fetch_url(url: str) -> Optional[bytes]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
            try:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return None

    @staticmethod
    def extract_from_pdf(file_obj: Union[str, io.BytesIO]) -> str:
        text = ""
        try:
            with pdfplumber.open(file_obj) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        return text

    @staticmethod
    def extract_from_html(html_content: bytes) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        text = soup.get_text()
        return text

class GeminiDistiller:
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        if not api_keys:
            raise ValueError("API Keys must be provided.")
        
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self._configure_current_key()
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.request_interval = 1.0

    def _configure_current_key(self):
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Switched to API Key #{self.current_key_index + 1}")

    def _rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()

    def _build_prompt(self, spanish_texts: List[str]) -> str:
        json_format = """
[
  {
    "es": "Original Spanish sentence",
    "nah": "Nahuatl translation",
    "myn": "Maya translation"
  }
]
"""
        # Truncate very long texts in the prompt if necessary, but batching handles this mostly.
        # Ensure we don't send massive blocks that confuse the JSON requirement.
        
        prompt = f"""
Act as a high-level linguist and polyglot expert in Indigenous languages of Mexico.
Your task is to translate the following Spanish text fragments into TWO languages:
1. Classical/Central Nahuatl (incorporating reverential forms where appropriate).
2. Yucatec Maya (modern standard).

If the input is a fragment or headline, translate it contextually.
Input Spanish Text:
{json.dumps(spanish_texts, ensure_ascii=False, indent=2)}

RETURN ONLY A VALID JSON ARRAY. No markdown formatting, no explanations.
Target Format:
{json_format}
"""
        return prompt

    async def distill_batch(self, batch: List[str], retries: int = 3) -> List[Dict]:
        prompt = self._build_prompt(batch)
        
        # Increased retries to account for key rotation attempts
        max_attempts = retries * len(self.api_keys) 
        
        for attempt in range(max_attempts):
            try:
                response = await self.model.generate_content_async(
                    prompt, 
                    safety_settings=self.safety_settings,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                text_response = response.text.strip()
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
                
                data = json.loads(text_response)
                # Ensure it's a list
                if isinstance(data, dict):
                    data = [data]
                return data

            except exceptions.ResourceExhausted:
                # If we have multiple keys, rotate immediately
                if len(self.api_keys) > 1:
                    logger.warning(f"Key #{self.current_key_index + 1} exhausted. Rotating...")
                    self._rotate_key()
                    # Don't sleep, just retry immediately with new key
                    continue
                else:
                    # Only one key, use standard backoff
                    wait_time = (2 ** (attempt % 3)) + self.request_interval
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Error processing batch {e}")
                if attempt == max_attempts - 1:
                    return []
                await asyncio.sleep(2)
        
        return []

    async def process_text_content(self, text: str, source_name: str, out_f, batch_size: int):
        """Helper to chunk and distill text content immediately."""
        raw_text = ContentExtractor.clean_text(text)
        chunks = ContentExtractor.chunk_text(raw_text)
        
        # Filter chunks
        chunks = [c for c in chunks if len(c) > 10]
        if not chunks: return

        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(f"Source {source_name}: {len(chunks)} chunks, {total_batches} batches.")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            results = await self.distill_batch(batch)
            if results:
                for item in results:
                    item['source'] = source_name # Track source
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
            await asyncio.sleep(self.request_interval)
            
    async def run_pipeline(self, input_path: str, output_file: str, batch_size: int = 5):
        logger.info(f"Starting distillation. Input: {input_path}")
        
        # Resume Logic
        processed_log = "data/silver/processed_sources.log"
        processed_files = set()
        if os.path.exists(processed_log):
             with open(processed_log, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f)

        # Output file handle
        # Append mode to keep previous results
        out_f = open(output_file, 'a', encoding='utf-8')
        log_f = open(processed_log, 'a', encoding='utf-8')

        try:
            # 1. Handling List of URLs/Files (The most common mass case)
            sources = []
            
            # Determine sources list
            if input_path.startswith("http"):
                sources = [input_path]
            elif os.path.exists(input_path):
                 # Check if it's a content file or list
                 is_list = False
                 if input_path.endswith('.txt'):
                     with open(input_path, 'r', encoding='utf-8') as f:
                         first = f.readline().strip()
                         if first.startswith("http") or os.path.exists(first):
                             is_list = True
                 
                 if is_list:
                     with open(input_path, 'r', encoding='utf-8') as f:
                         sources = [l.strip() for l in f if l.strip()]
                 else:
                     # Treat input_path as the single source file
                     sources = [input_path]
            
            # 2. Process Loop
            for idx, source in enumerate(sources):
                if source in processed_files:
                    logger.info(f"Skipping done: {source}")
                    continue
                
                logger.info(f"Processing ({idx+1}/{len(sources)}): {source}")
                
                try:
                    content_text = ""
                    if source.startswith("http"):
                        content = await ContentExtractor.fetch_url(source)
                        if content:
                            if source.lower().endswith(".pdf") or content.startswith(b"%PDF"):
                                content_text = ContentExtractor.extract_from_pdf(io.BytesIO(content))
                            else:
                                content_text = ContentExtractor.extract_from_html(content)
                    elif os.path.exists(source):
                        if source.lower().endswith(".pdf"):
                            content_text = ContentExtractor.extract_from_pdf(source)
                        elif source.lower().endswith(".jsonl"):
                             # Handle JSONL - read all lines
                             lines = []
                             with open(source, 'r', encoding='utf-8') as jf:
                                 for line in jf:
                                     if line.strip(): lines.append(json.loads(line).get('es',''))
                             content_text = " ".join(lines)
                        else:
                            with open(source, 'r', encoding='utf-8') as tf:
                                content_text = tf.read()
                    
                    if content_text:
                        await self.process_text_content(content_text, source, out_f, batch_size)
                        
                        # Mark Done
                        log_f.write(source + "\n")
                        log_f.flush()
                    else:
                        logger.warning(f"Empty content for {source}")

                except Exception as e:
                    logger.error(f"Error processing {source}: {e}")
                    # Don't mark as done so we retry later
                    
        finally:
            out_f.close()
            log_f.close()
            logger.info("Pipeline run finished.")

def main():
    parser = argparse.ArgumentParser(description="Gemini Distiller for Nahuatl/Maya (Web/PDF/Text)")
    parser.add_argument("--input", required=True, help="Input file path or URL")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for API calls")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"), help="Google API Key")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling API (Test mode)")
    
    args = parser.parse_args()

    if args.dry_run:
        logger.info(f"[DRY RUN] Would process input: {args.input}")
        # Actually run the extraction logic to verify it works, but mock the API call
        distiller = GeminiDistiller(api_key="dry_run_key")
        
        # Monkey patch distill_batch for dry run
        async def mock_distill(batch, retries=3):
            logger.info(f"[DRY RUN] Distilling batch of size {len(batch)}")
            return [{"es": s, "nah": "[MOCK NAHUATL]", "myn": "[MOCK MAYA]"} for s in batch]
        
        distiller.distill_batch = mock_distill
        asyncio.run(distiller.run_pipeline(args.input, args.output, args.batch_size))
        return

    # Parse Keys: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3...
    api_keys = []
    
    # Primary Key
    primary_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if primary_key:
        api_keys.append(primary_key)
    
    # Secondary Keys (Environment only)
    for i in range(2, 10): # Check up to GEMINI_API_KEY_9
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if key:
            api_keys.append(key)

    if not api_keys:
        logger.error("No API Keys found. Set GOOGLE_API_KEY or GEMINI_API_KEY env vars.")
        return
        
    logger.info(f"Loaded {len(api_keys)} API Key(s). Rotation enabled.")

    distiller = GeminiDistiller(api_keys=api_keys)
    asyncio.run(distiller.run_pipeline(args.input, args.output, args.batch_size))

if __name__ == "__main__":
    main()
