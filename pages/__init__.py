"""
Pages package for FAOSTAT Analytics Application.

This package contains all the main page components for the Streamlit application,
including home page, dataset browser, analysis pages, and specialized interfaces.
"""

from pages.home import show_home_page
from pages.dataset_browser import show_dataset_browser
from pages.analysis import show_analysis_page
from pages.thematic_templates import show_thematic_templates_page
from pages.nl_query import show_nl_query_page
from pages.about import show_about_page

__all__ = [
    'show_home_page',
    'show_dataset_browser',
    'show_analysis_page', 
    'show_thematic_templates_page',
    'show_nl_query_page',
    'show_about_page'
]