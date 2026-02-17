import pandas as pd
import glob
import os

def convert_jsonl_to_parquet(directory="data/gold"):
    files = glob.glob(os.path.join(directory, "*.jsonl"))
    print(f"Found {len(files)} JSONL files in {directory}...")
    
    for f in files:
        try:
            print(f"Converting {os.path.basename(f)} to Parquet...")
            df = pd.read_json(f, lines=True)
            output_name = f.replace(".jsonl", ".parquet")
            df.to_parquet(output_name, index=False)
            print(f"✅ Saved {output_name}")
        except Exception as e:
            print(f"❌ Failed to convert {f}: {e}")

if __name__ == "__main__":
    convert_jsonl_to_parquet()
