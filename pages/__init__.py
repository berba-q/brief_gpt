"""
Pages package for FAOSTAT Analytics Application.

This package contains all the main page components for the Streamlit application,
including home page, dataset browser, analysis pages, and specialized interfaces.
"""

from pages.home import show_home_page
from pages.dataset_browser import show_dataset_browser
from pages.analysis import show_analysis_page
from components.visualization_gallery import render_visualization_gallery
from pages.thematic_templates import show_template_preview
from pages.about import show_about_page

__all__ = [
    'show_home_page',
    'show_dataset_browser',
    'show_analysis_page',
    'render_visualization_gallery',
    'show_template_preview',
    'show_about_page'
]