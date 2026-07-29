"""
Download PROMISE Repository CK metric datasets.
Source: https://github.com/klainfo/DefectData
"""

import urllib.request
import os

BASE_URL = "https://raw.githubusercontent.com/klainfo/DefectData/master/inst/extdata/terapromise/ck"

DATASETS = [
    "ant-1.7.csv", "camel-1.6.csv", "ivy-2.0.csv", "jedit-4.3.csv",
    "log4j-1.2.csv", "lucene-2.4.csv", "poi-3.0.csv", "synapse-1.2.csv",
    "velocity-1.6.csv", "xalan-2.7.csv", "xerces-1.4.csv",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, "datasets")
os.makedirs(SAVE_DIR, exist_ok=True)

success, failed = 0, 0

for filename in DATASETS:
    url  = f"{BASE_URL}/{filename}"
    dest = os.path.join(SAVE_DIR, filename)
    if os.path.exists(dest):
        print(f"  Already exists: {filename}")
        success += 1
        continue
    print(f"Downloading {filename} ...", end=" ")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        print("OK")
        success += 1
    except Exception as e:
        print(f"FAILED — {e}")
        failed += 1

print(f"\nDone. {success} downloaded, {failed} failed.")
print(f"Files saved in: {SAVE_DIR}")
