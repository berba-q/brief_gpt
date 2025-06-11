
"""
Features package for FAOSTAT Analytics Application.

This package contains advanced features and functionality modules including
bookmarking system, template management, and natural language query processing.
"""

from features.bookmarks import BookmarkManager
from features.template_manager import TemplateManager
from features.nl_query_engine import NaturalLanguageQueryEngine

__all__ = [
    'BookmarkManager',
    'TemplateManager', 
    'NaturalLanguageQueryEngine'
]