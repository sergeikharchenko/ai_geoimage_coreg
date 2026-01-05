"""
AI-GeoImage-Coreg: AI-based Georeferencing Package
Exports the main pipeline and CLI entry point.
"""

from .core import run_pipeline, main_cli

# Version of the package
__version__ = "0.1.0"
__author__ = "Sergei Kharchenko"

# Define what is available when running 'from AI-GeoImage-Coreg import *'
__all__ = ["run_pipeline", "main_cli"]
