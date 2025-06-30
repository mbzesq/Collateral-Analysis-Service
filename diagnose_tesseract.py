#!/usr/bin/env python3
"""
Diagnostic script to verify Tesseract installation and configuration
"""
import os
import subprocess
import glob

print("=== Tesseract Diagnostic Report ===\n")

# Check Tesseract binary
print("1. Tesseract Binary:")
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract']
for path in tesseract_paths:
    if os.path.exists(path):
        print(f"   ✓ Found at: {path}")
        # Get version
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True)
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
        except:
            pass
    else:
        print(f"   ✗ Not found at: {path}")

# Check common tessdata locations
print("\n2. Tessdata Directories:")
tessdata_paths = [
    '/usr/share/tesseract-ocr/tessdata',
    '/usr/share/tesseract-ocr/4.00/tessdata',
    '/usr/share/tesseract-ocr/5.00/tessdata',
    '/usr/local/share/tessdata',
    '/usr/share/tessdata'
]

found_tessdata = False
for path in tessdata_paths:
    if os.path.exists(path):
        print(f"   ✓ Found: {path}")
        # List language files
        lang_files = glob.glob(os.path.join(path, '*.traineddata'))
        if lang_files:
            print(f"     Languages: {', '.join([os.path.basename(f).replace('.traineddata', '') for f in lang_files[:5]])}")
            if len(lang_files) > 5:
                print(f"     ... and {len(lang_files) - 5} more")
        found_tessdata = True
    else:
        print(f"   ✗ Not found: {path}")

# Check environment variables
print("\n3. Environment Variables:")
env_vars = ['TESSDATA_PREFIX', 'TESSERACT_PREFIX']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        print(f"   {var} = {value}")
        if os.path.exists(value):
            print(f"     ✓ Path exists")
        else:
            print(f"     ✗ Path does NOT exist")
    else:
        print(f"   {var} is not set")

# Find all tessdata directories
print("\n4. All tessdata directories found:")
try:
    result = subprocess.run(['find', '/', '-name', 'tessdata', '-type', 'd', '2>/dev/null'], 
                          capture_output=True, text=True, shell=True)
    tessdata_dirs = [d for d in result.stdout.strip().split('\n') if d and 'proc' not in d]
    for d in tessdata_dirs[:10]:  # Limit output
        print(f"   - {d}")
        # Check for eng.traineddata
        eng_file = os.path.join(d, 'eng.traineddata')
        if os.path.exists(eng_file):
            print(f"     ✓ Contains eng.traineddata")
except:
    print("   Error running find command")

# Test Tesseract
print("\n5. Tesseract Test:")
try:
    import pytesseract
    print(f"   pytesseract module loaded successfully")
    print(f"   pytesseract.tesseract_cmd = {pytesseract.pytesseract.tesseract_cmd}")
    
    # Try to run tesseract
    from PIL import Image
    import numpy as np
    
    # Create a simple test image with text
    img = Image.new('RGB', (100, 30), color='white')
    try:
        text = pytesseract.image_to_string(img)
        print(f"   ✓ Tesseract OCR test successful")
    except Exception as e:
        print(f"   ✗ Tesseract OCR test failed: {str(e)}")
except ImportError as e:
    print(f"   ✗ Failed to import required module: {e}")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

print("\n=== End of Diagnostic Report ===")