"""
Pages package for FAOSTAT Analytics Application.

This package contains all the main page components for the Streamlit application,
including home page, dataset browser, analysis pages, and specialized interfaces.
"""

from pages_backup.home import show_home_page
from pages_backup.dataset_browser import show_dataset_browser
from pages_backup.analysis import show_analysis_page
from pages_backup.about import show_about_page
from pages_backup.nl_query import show_nl_query_page
from pages_backup.thematic_templates import show_thematic_templates_page

# Import visualization gallery from components (not pages)
from components.visualization_gallery import render_visualization_gallery

__all__ = [
    'show_home_page',
    'show_dataset_browser', 
    'show_analysis_page',
    'show_about_page',
    'show_nl_query_page',
    'show_thematic_templates_page',
    'render_visualization_gallery'
]