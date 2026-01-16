
import sqlite3
import json
from pathlib import Path

def check_redundancy():
    db_path = Path("data/metadata.db")
    if not db_path.exists():
        print("Metadata DB not found.")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT chunk_id, json_data FROM metadata LIMIT 1")
        row = cursor.fetchone()
        cursor.execute("SELECT chunk_id, json_data FROM metadata")
        
        print(f"Inspecting Metadata DB: {db_path} (Size: {db_path.stat().st_size} bytes)")
    
        rows = cursor.fetchall()
        print(f"Total rows in metadata: {len(rows)}")
        
        clean_chunks = 0
        dirty_chunks = 0
        
        for row in rows:
            chunk_id, json_str = row
            data = json.loads(json_str)
            
            has_issue = False
            if 'content' in data:
                has_issue = True
            if 'embedding' in data:
                has_issue = True
                
            if has_issue:
                dirty_chunks += 1
                print(f"[FAIL] Dirty Chunk: {chunk_id}. Keys: {list(data.keys())}")
            else:
                clean_chunks += 1
                
        print(f"\nSummary: {clean_chunks} clean, {dirty_chunks} dirty chunks.")

    finally:
        conn.close()

if __name__ == "__main__":
    check_redundancy()
