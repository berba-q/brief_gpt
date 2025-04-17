"""
Configuration manager for FAOSTAT Analytics Application.

This module provides a class for loading, accessing, and saving configuration
from YAML files. It handles nested configuration properties using dot notation
and provides type-safe access to configuration values.

This is done to prevent the need to use a global configuration object throughout the
application, which can lead to issues with state management and testing.
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path

from config.constants import DEFAULT_CONFIG_PATH, DEFAULT_CONFIG

# Set up logger
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager that reads from YAML files with dot notation access.
    
    This class provides a flexible way to manage application configuration:
    - Loads configuration from YAML files
    - Provides access to nested properties using dot notation
    - Allows setting and saving configuration values
    - Supports default values for missing properties
    
    Attributes:
        config_path (str): Path to the YAML configuration file
        config (dict): The loaded configuration dictionary
    """
    
    # Function to load default configuration
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize the configuration manager
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path # Default path for configuration
        self.config = {}
        
        # Load configuration
        self.load_config()
    
    # Load configuration from file
    def load_config(self) -> None:
        """
        Load configuration from YAML file
        
        If the configuration file doesn't exist, it will create a default one.
        If loading fails, it will use empty configuration.
        """
        # Start with default config
        self.config = DEFAULT_CONFIG.copy()
        
        # Check if config file exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                    # Deep merge loaded config with default config
                    self._deep_merge(self.config, loaded_config)
            except Exception as e:
                logger.error(f"Error loading configuration file: {str(e)}")
        else:
            # Create default config file if it doesn't exist
            logger.info(f"Configuration file not found. Creating default at {self.config_path}")
            self._ensure_config_dir()
            self.save()
    
    # Function to ensure the directory for the configuration file exists
    def _ensure_config_dir(self) -> None:
        """Ensure the directory for the configuration file exists"""
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
    
    # Function to deep merge two dictionaries, modifies the base dictionary
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, modifying base
        
        Args:
            base: Base dictionary to be updated
            update: Dictionary with updates to apply
            
        Returns:
            The updated base dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    # Function to get a configuration value using dot notation. Allows for easy nested access of properties
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key: Configuration key (can be nested with dots, e.g. 'api_keys.openai')
            default: Default value to return if key not found
            
        Returns:
            The configuration value or default
        """
        # Split the key by dots for nested access
        parts = key.split('.')
        
        # Navigate to the value
        value = self.config
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        
        return value
    
    # Function to get an integer configuration value, also handles conversion and defaulting
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get an integer configuration value
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value to return if key not found or not an integer
            
        Returns:
            The integer configuration value or default
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    # Function to get a float configuration value, also handles conversion and defaulting
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get a float configuration value
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value to return if key not found or not a float
            
        Returns:
            The float configuration value or default
        """
        value = self.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    # Function to get a boolean configuration value, also handles conversion and defaulting
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get a boolean configuration value
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value to return if key not found or not a boolean
            
        Returns:
            The boolean configuration value or default
        """
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return default
    
    # Function to get a list configuration value, also handles conversion and defaulting
    def get_list(self, key: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
        """
        Get a list configuration value, also handles conversion and defaulting
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value to return if key not found or not a list
            
        Returns:
            The list configuration value or default
        """
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        return default
    
    # Function to set a configuration value using dot notation
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation
        
        Args:
            key: Configuration key (can be nested with dots)
            value: Value to set
        """
        # Split the key by dots for nested access
        parts = key.split('.')
        
        # Navigate to the right place in the config
        config = self.config
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last part, set the value
                config[part] = value
            else:
                # Create dict if it doesn't exist
                if part not in config or not isinstance(config[part], dict):
                    config[part] = {}
                config = config[part]
    
    # Function to save the configuration to a YAML file
    def save(self) -> bool:
        """
        Save configuration to YAML file
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            self._ensure_config_dir()
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    # Function to get the OpenAI API key from configuration
    def get_openai_api_key(self) -> Optional[str]:
        """
        Get the OpenAI API key from configuration
        
        Returns:
            str: The OpenAI API key or None
        """
        return self.get('api_keys.openai')
    
    # Function to set the OpenAI API key in configuration
    def set_openai_api_key(self, api_key: str) -> None:
        """
        Set the OpenAI API key in configuration
        
        Args:
            api_key: The OpenAI API key to set
        """
        self.set('api_keys.openai', api_key)
    
    # Function to get the output directory from configuration, this is where the application will save its output files
    def get_output_directory(self) -> str:
        """
        Get the output directory from configuration
        
        Returns:
            str: The output directory
        """
        return self.get('app.output_directory', 'output')
    
    # Function to get the FAOSTAT base URL from configuration to access all of FAOSTAT data
    def get_faostat_base_url(self) -> str:
        """
        Get the FAOSTAT base URL from configuration
        
        Returns:
            str: The FAOSTAT base URL
        """
        return self.get('faostat.base_url', 'https://bulks-faostat.fao.org/production/')
    
    # Function to get model configuration for OpenAI API
    def get_model_config(self, model_type: str = 'default') -> Dict[str, Any]:
        """
        Get model configuration
        
        Args:
            model_type: Type of model (default, comprehensive, standard, etc.)
            
        Returns:
            dict: Model configuration
        """
        if model_type == 'default':
            return {
                'model': self.get('openai.default_model', 'gpt-4'),
                'max_tokens': self.get_int('openai.max_tokens', 1500),
                'temperature': self.get_float('openai.temperature', 0.7)
            }
        
        # Get specific model configuration
        model_config = self.get(f'openai.models.{model_type}', {})
        if not model_config:
            return self.get_model_config()  # Return default if specific not found
        
        return model_config