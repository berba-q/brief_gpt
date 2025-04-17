"""
Data models for FAOSTAT Analytics Application.

This module provides data classes and structures for working with FAOSTAT data,
ensuring type safety and consistent data handling across the application.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

@dataclass
class DatasetMetadata:
    """Metadata for a FAOSTAT dataset."""
    
    code: str
    name: str
    last_updated: str
    description: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisFilter:
    """Filter criteria for dataset analysis."""
    
    year_range: Optional[tuple[int, int]] = None
    regions: Optional[List[str]] = None
    items: Optional[List[str]] = None
    elements: Optional[List[str]] = None

@dataclass
class VisualizationInfo:
    """Information about a generated visualization."""
    
    type: str
    title: str
    filename: str
    filepath: str
    description: str
    data_source: Optional[str] = None

@dataclass
class AnalysisResults:
    """Results of a dataset analysis."""
    
    dataset_code: str
    dataset_name: str
    summary_stats: Dict[str, Any]
    figures: List[VisualizationInfo]
    filter_options: AnalysisFilter
    insights: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ThematicTemplate:
    """Template for thematic analysis."""
    
    id: str
    name: str
    description: str
    datasets: List[Dict[str, str]]
    key_indicators: List[str]
    regions_focus: List[str]
    default_visualizations: List[str]
    prompt_template: str

@dataclass
class Bookmark:
    """Saved analysis bookmark."""
    
    id: str
    name: str
    dataset_code: str
    dataset_name: str
    filters: AnalysisFilter
    created_at: str
    description: Optional[str] = None

@dataclass
class NaturalLanguageQuery:
    """Structured representation of a natural language query."""
    
    original_query: str
    dataset_code: str
    years: Optional[List[int]] = None
    regions: Optional[List[str]] = None
    items: Optional[List[str]] = None
    elements: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    visualization_type: Optional[str] = None