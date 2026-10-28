from __future__ import annotations

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent / "fastapi_qlora"
if str(PKG_DIR) not in sys.path:
	sys.path.insert(0, str(PKG_DIR))

from cli_chat import main

if __name__ == "__main__":
	main()
