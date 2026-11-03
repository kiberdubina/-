import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))

from app.main import main

main()
