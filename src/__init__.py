"""WebSearch QA System Package."""

from .main import WebSearchQASystem

__version__ = "0.1.0"
__all__ = ["WebSearchQASystem"]


# src/core/__init__.py
"""Core module for WebSearch QA System."""
from .models import *
from .pipeline import QAPipeline

__all__ = ["QAPipeline"]
