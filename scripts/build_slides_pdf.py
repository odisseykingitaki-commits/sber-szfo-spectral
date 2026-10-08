"""Thin wrapper → build_jury_slides (canonical 11-slide PDF builder)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_jury_slides import build

if __name__ == "__main__":
    build()
