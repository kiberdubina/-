import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))

from app.main import main

# Контрольный прогон считает камеры в дочерних процессах. При запуске процесса
# Python заново импортирует этот файл — окно должно открываться только в главном.
if __name__ == "__main__":
    main()
