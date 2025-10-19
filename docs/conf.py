"""Sphinx configuration for Agentic Context documentation."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

project = "Agentic Context Toolkit"
author = "lioarce01"
copyright = f"{datetime.utcnow():%Y}, {author}"

try:
    release = pkg_version("acet")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = ["myst_parser"]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_static_path: list[str] = []

source_suffix = ".md"
master_doc = "index"
