"""
Module 2.3 — Configuration Schema and Logging

Shared YAML configuration loader and structured evaluation logger.
No dependencies on other ooc_optimizer modules.

Public API:
    load_config(path) -> dict
    EvaluationLogger(log_path)
"""

from ooc_optimizer.config.schema import load_config
from ooc_optimizer.config.logger import EvaluationLogger

__all__ = ["load_config", "EvaluationLogger"]
