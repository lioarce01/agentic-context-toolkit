"""Sphinx configuration for Agentic Context documentation."""

from __future__ import annotations

import datetime
from importlib.metadata import PackageNotFoundError, version

project = "Agentic Context"
author = "Agentic Context maintainers"
copyright = f"{datetime.datetime.utcnow():%Y}, {author}"

try:
    release = version("act")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions: list[str] = []

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_static_path: list[str] = []

source_suffix = ".md"
master_doc = "index"
