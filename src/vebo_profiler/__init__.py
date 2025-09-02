"""
Vebo Data Profiler - Python Code Generation System

This package provides automatic data profiling capabilities by generating
Python code from rules and executing checks against datasets.
"""

__version__ = "0.1.0"
__author__ = "Vebo Team"

from .core.profiler import VeboProfiler
from .core.rule_engine import RuleEngine
from .core.check_executor import CheckExecutor

__all__ = ["VeboProfiler", "RuleEngine", "CheckExecutor"]
