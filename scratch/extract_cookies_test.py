import os
import json
import sqlite3
import shutil
import base64
import win32crypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def main():
    db_path = os.path.expandvars(r"%USERPROFILE%\AppData\Local\Google\Chrome\User Data\Default\Network\Cookies")
    if not os.path.exists(db_path):
        print(f"Cookies file not found at: {db_path}")
        return
        
    temp_db = "temp_cookies.db"
    shutil.copyfile(db_path, temp_db)
    
    conn = None
    try:
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        query = """
        SELECT host_key, name, encrypted_value 
        FROM cookies 
        WHERE host_key LIKE '%labs.google%'
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for host_key, name, encrypted_value in rows:
            prefix = encrypted_value[:4]
            print(f"Cookie: {name} (domain: {host_key}), len={len(encrypted_value)}, prefix={prefix}")
                
    finally:
        if conn:
            conn.close()
        if os.path.exists(temp_db):
            os.remove(temp_db)

if __name__ == "__main__":
    main()
