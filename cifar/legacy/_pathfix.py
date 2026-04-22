from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
