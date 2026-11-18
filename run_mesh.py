"""Local launcher for arena-lite command mode: python ../run_mesh.py (the image runs `python -u -m mesh`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh.main import main  # noqa: E402

main()
