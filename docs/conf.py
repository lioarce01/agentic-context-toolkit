"""Sphinx configuration for Agentic Context documentation."""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for autodoc (future use)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Agentic Context Toolkit"
author = "Lioarce01"
copyright = f"{datetime.datetime.utcnow():%Y}, {author}"

try:
    release = importlib.metadata.version("act")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = release

extensions: list[str] = []

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_static_path = ["_static"]

