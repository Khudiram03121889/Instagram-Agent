import json
import time

def normalize_cookie(c):
    """Convert extension cookie format to Playwright format."""
    out = {}
    # Mandatory fields
    out['name'] = c.get('name', '')
    out['value'] = c.get('value', '')
    out['domain'] = c.get('domain', '')
    out['path'] = c.get('path', '/')
    
    # Expiration
    if 'expirationDate' in c:
        out['expires'] = c['expirationDate']
    elif 'expires' in c:
        out['expires'] = c['expires']
    else:
        out['expires'] = time.time() + 31536000 # Default 1 year
        
    # Flags
    out['httpOnly'] = c.get('httpOnly', False)
    out['secure'] = c.get('secure', False)
    
    # SameSite mapping
    ss = c.get('sameSite', 'None')
    if ss in ['no_restriction', 'unspecified']:
        ss = 'None'
    out['sameSite'] = ss.capitalize() # Playwright expects Lax, Strict, None
    
    if out['sameSite'] not in ['Lax', 'Strict', 'None']:
        out['sameSite'] = 'None'
        
    return out

def main():
    print(f"\n{'='*60}")
    print("🍪 MANUAL COOKIE IMPORT (Bypass Automation Blocks)")
    print(f"{'='*60}")
    print("Since your computer blocks the automation port, we will copy the keys manually.")
    print("\n👉 INSTRUCTIONS:")
    print("1. Open your Chrome normally.")
    print("2. Install the 'EditThisCookie' extension (or any Cookie Editor that exports JSON).")
    print("3. Go to https://labs.google.com/fx/tools/flow")
    print("4. Use the extension to EXPORT cookies to Clipboard (JSON format).")
    print("5. Paste the content below.")
    print("6. Press ENTER, then press Ctrl+Z (Windows) or Ctrl+D (Linux/Mac) and ENTER again to finish.")
    print(f"{'-'*60}")
    print("PASTE JSON BELOW:")
    
    try:
        # Read until EOF
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        
        raw_text = "\n".join(lines).strip()
        if not raw_text:
            print("❌ No data pasted.")
            return

        # Handle case where user pastes just the list []
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            print("❌ Invalid JSON format. Make sure you copied the whole export.")
            return
            
        if not isinstance(data, list):
            print("❌ Expected a list of cookies (starts with [).")
            return
            
        print(f"   Processing {len(data)} cookies...")
        
        # Convert
        playwright_cookies = [normalize_cookie(c) for c in data]
        
        storage_state = {
            "cookies": playwright_cookies,
            "origins": [] # LocalStorage is usually optional for auth
        }
        
        with open("auth.json", "w") as f:
            json.dump(storage_state, f, indent=2)
            
        print(f"\n✅ SUCCESS! Saved {len(playwright_cookies)} cookies to 'auth.json'")
        print("🎉 You can now run 'python main.py'")
        
    except KeyboardInterrupt:
        print("\n❌ Cancelled.")

if __name__ == "__main__":
    main()
