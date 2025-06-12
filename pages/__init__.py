"""
Pages package for FAOSTAT Analytics Application.

This package contains all the main page components for the Streamlit application,
including home page, dataset browser, analysis pages, and specialized interfaces.
"""

from pages.home import show_home_page
from pages.dataset_browser import render
from pages.analysis import render
from components.visualization_gallery import render_visualization_gallery
from pages.thematic_templates import render
from pages.about import render

__all__ = [
    'show_home_page',
    'render',
    'render', 
    'render_visualization_gallery',
    'render',
    'render',
    'render'
]