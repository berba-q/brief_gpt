"""
Constants for FAOSTAT Analytics Application configuration.

This module provides default paths, values, and configuration structures 
used throughout the application.
"""

import os
from pathlib import Path

# Configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yml")

# Default FAOSTAT API URL
DEFAULT_FAOSTAT_URL = "https://bulks-faostat.fao.org/production/"
DEFAULT_DATASETS_ENDPOINT = "datasets_E.json"

# Default cache duration (in seconds)
DEFAULT_CACHE_DURATION = 3600  # 1 hour

# Default document generation settings
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "figures")
DEFAULT_REPORTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "reports")
DEFAULT_BOOKMARKS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "bookmarks")

# Default OpenAI settings
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.7

# Application theme
DEFAULT_THEME = "light"

# Default visualization settings
DEFAULT_MAX_FIGURES = 8
DEFAULT_FIGURE_WIDTH = 10
DEFAULT_FIGURE_HEIGHT = 6
DEFAULT_DPI = 300

# Default color palette
DEFAULT_COLOR_PALETTE = {
    "primary": "#3498db",    # Blue
    "secondary": "#2ecc71",  # Green
    "tertiary": "#e74c3c",   # Red
    "neutral": "#95a5a6",    # Gray
    "highlight": "#f39c12"   # Orange
}

# Default region colors
DEFAULT_REGION_COLORS = {
    "Africa": "#3498db",      # Blue
    "Americas": "#e74c3c",    # Red
    "Asia": "#2ecc71",        # Green
    "Europe": "#9b59b6",      # Purple
    "Oceania": "#f39c12"      # Orange
}

# Default nutrient colors for fertilizer data
DEFAULT_NUTRIENT_COLORS = {
    "Nitrogen": "#dd4b39",    # Red
    "Phosphate": "#28a745",   # Green
    "Potash": "#ffc107"       # Yellow
}

# Default country ranking comparison colors
DEFAULT_RANKING_COLORS = {
    "top": "#28a745",         # Green for top performers
    "bottom": "#dd4b39"       # Red for bottom performers
}

# Default configuration structure
DEFAULT_CONFIG = {
    "api_keys": {
        "openai": ""  # Will be populated via UI or YAML file
    },
    "faostat": {
        "base_url": DEFAULT_FAOSTAT_URL,
        "datasets_endpoint": DEFAULT_DATASETS_ENDPOINT,
        "cache_duration": DEFAULT_CACHE_DURATION
    },
    "app": {
        "output_directory": DEFAULT_OUTPUT_DIR,
        "theme": DEFAULT_THEME,
        "default_language": "en",
        "max_figures_per_report": DEFAULT_MAX_FIGURES,
        "default_report_format": "pdf"
    },
    "openai": {
        "default_model": DEFAULT_OPENAI_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "models": {
            "comprehensive": {
                "model": "gpt-4",
                "max_tokens": 1500,
                "temperature": 0.7
            },
            "standard": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.5
            }
        }
    },
    "ui": {
        "sidebar_width": "medium",
        "show_debug_info": False,
        "color_palette": DEFAULT_COLOR_PALETTE
    },
    "visualization": {
        "figure_width": DEFAULT_FIGURE_WIDTH,
        "figure_height": DEFAULT_FIGURE_HEIGHT,
        "dpi": DEFAULT_DPI,
        "region_colors": DEFAULT_REGION_COLORS,
        "nutrient_colors": DEFAULT_NUTRIENT_COLORS,
        "ranking_colors": DEFAULT_RANKING_COLORS
    }
}

# Thematic templates configuration
THEMATIC_TEMPLATES = {
    "food_security": {
        "name": "Food Security Analysis",
        "description": "Analyze food availability, access, and stability across countries and regions",
        "datasets": [
            {"code": "FBS", "name": "Food Balances"},
            {"code": "FS", "name": "Suite of Food Security Indicators"},
            {"code": "FTL", "name": "Food Loss"}
        ],
        "key_indicators": [
            "Dietary Energy Supply", 
            "Prevalence of Undernourishment",
            "Food Loss Percentage"
        ],
        "regions_focus": ["Africa", "Asia", "Small Island Developing States"],
        "default_visualizations": ["time_series", "regional_comparison", "item_comparison"],
        "prompt_template": """Analyze food security trends in the selected regions with focus on:
1. Changes in dietary energy supply
2. Undernourishment prevalence patterns
3. Relationship between food loss and security indicators
4. Policy implications based on the data"""
    },
    "climate_impact": {
        "name": "Climate Impact on Agriculture",
        "description": "Examine relationships between emissions, land use, and agricultural productivity",
        "datasets": [
            {"code": "GT", "name": "Greenhouse Gas Emissions"}, 
            {"code": "EF", "name": "Emissions Intensities"}, 
            {"code": "QCL", "name": "Crops and Livestock Products"}
        ],
        "key_indicators": [
            "Emissions (CO2eq)", 
            "Crop Yield", 
            "Land Use Change"
        ],
        "regions_focus": ["Global", "By continent", "Top 10 emitters"],
        "default_visualizations": ["correlation_chart", "time_series", "heatmap"],
        "prompt_template": """Analyze the relationship between agricultural emissions and productivity with focus on:
1. Trends in emissions vs. agricultural output
2. Regional differences in emissions intensity
3. Changes in efficiency over time
4. Potential mitigation strategies based on data trends"""
    },
    "fertilizer_analysis": {
        "name": "Fertilizer Use and Impacts",
        "description": "Explore patterns in fertilizer application, efficiency, and environmental impacts",
        "datasets": [
            {"code": "RFN", "name": "Fertilizers by Nutrient"},
            {"code": "EF", "name": "Emissions Factors"},
            {"code": "QCL", "name": "Crops and Livestock Products"}
        ],
        "key_indicators": [
            "Nitrogen Application",
            "Phosphorus Application", 
            "Potassium Application",
            "Crop Yield",
            "Emissions"
        ],
        "regions_focus": ["By continent", "Top/Bottom 10 users"],
        "default_visualizations": ["time_series", "correlation_chart", "top10_comparison"],
        "prompt_template": """Analyze fertilizer use patterns and impacts with focus on:
1. Trends in application rates by nutrient type
2. Relationship between fertilizer use and crop yields
3. Environmental implications based on emissions data
4. Regional differences in fertilizer efficiency"""
    },
    "agricultural_trade": {
        "name": "Agricultural Trade Analysis",
        "description": "Examine import/export patterns and trade balances for agricultural products",
        "datasets": [
            {"code": "TCL", "name": "Crops and Livestock Products Trade"},
            {"code": "TM", "name": "Detailed Trade Matrix"}
        ],
        "key_indicators": [
            "Export Value",
            "Import Value",
            "Trade Balance",
            "Trade Volume"
        ],
        "regions_focus": ["Major exporters", "Major importers", "Regional blocs"],
        "default_visualizations": ["flow_chart", "time_series", "balance_chart"],
        "prompt_template": """Analyze agricultural trade patterns with focus on:
1. Major exporters and importers for key commodities
2. Changes in trade balances over time
3. Regional trade dependencies
4. Impact of trade on local agricultural systems"""
    }
}