"""Run new-package tests without requiring an editable environment install."""
from pathlib import Path
import os
import sys
SOURCE = str(Path(__file__).resolve().parents[1]/'src_final')
sys.path.insert(0,SOURCE)
os.environ['PYTHONPATH'] = SOURCE + os.pathsep + os.environ.get('PYTHONPATH','')
