"""Puts <repo>/backend on sys.path so tests can `import app...` from any working directory."""
import os
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
