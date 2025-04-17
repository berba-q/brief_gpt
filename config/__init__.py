"""
Configuration management package for the brief generation application.

This package provides functionality for loading, storing, and managing
application configuration from YAML files.
"""

from config.config_manager import ConfigManager
from config.constants import *

__all__ = ['ConfigManager', 'DEFAULT_CONFIG_PATH']