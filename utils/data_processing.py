"""
Focus on essential functions needed for analytical brief generation:
- Basic data cleaning
- Top/bottom performer identification  
- Growth rate calculations
- Simple aggregations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on a FAOSTAT dataset.
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle column names - strip whitespace
    df.columns = [col.strip() for col in df.columns]
    
    # Convert common FAOSTAT columns to appropriate types
    type_conversions = {
        'Year': 'int64',
        'Area Code': 'int64', 
        'Item Code': 'int64',
        'Element Code': 'int64',
        'Value': 'float64'
    }
    
    for col, dtype in type_conversions.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if dtype == 'int64':
                    # Handle NaN in integer columns by converting to nullable int
                    df[col] = df[col].astype('Int64')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to {dtype}: {str(e)}")
    
    # Fill NaN values in Value column with 0 (common for FAOSTAT)
    if 'Value' in df.columns:
        df['Value'] = df['Value'].fillna(0)
    
    # Fill NaN values in string columns with empty string
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].fillna('')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def get_top_bottom_performers(
    df: pd.DataFrame,
    group_by: str,
    value_column: str = 'Value',
    top_n: int = 10,
    bottom_n: int = 10,
    year_filter: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify top and bottom performing areas/items.
    Perfect for your top/bottom N country selection.
    
    Args:
        df: Dataset DataFrame
        group_by: Column to group by (e.g., 'Area', 'Item')
        value_column: Column with values to compare
        top_n: Number of top performers to return
        bottom_n: Number of bottom performers to return
        year_filter: Optional year to filter the data
        
    Returns:
        Tuple of (top_performers_df, bottom_performers_df)
    """
    if df is None or df.empty or group_by not in df.columns or value_column not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Apply year filter if provided
    filtered_df = df.copy()
    if year_filter is not None and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Year'] == year_filter]
    
    # Aggregate data by the grouping column
    aggregated = filtered_df.groupby(group_by)[value_column].sum().reset_index()
    
    # Sort by value
    aggregated = aggregated.sort_values(value_column, ascending=False)
    
    # Get top performers
    top_performers = aggregated.head(top_n).copy()
    top_performers['rank'] = range(1, len(top_performers) + 1)
    
    # Get bottom performers  
    bottom_performers = aggregated.tail(bottom_n).copy()
    bottom_performers = bottom_performers.sort_values(value_column, ascending=True)
    bottom_performers['rank'] = range(1, len(bottom_performers) + 1)
    
    return top_performers, bottom_performers

def calculate_year_over_year_change(
    df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    time_column: str = 'Year',
    value_column: str = 'Value'
) -> pd.DataFrame:
    """
    Calculate year-over-year changes for analytical briefs.
    
    Args:
        df: Dataset DataFrame
        group_by: Optional list of columns to group by (e.g., ['Area', 'Item'])
        time_column: Column containing time periods
        value_column: Column containing values
        
    Returns:
        DataFrame with year-over-year changes
    """
    if df is None or df.empty or time_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()
    
    # Make a copy and sort by time
    result_df = df.copy()
    
    if group_by and all(col in result_df.columns for col in group_by):
        # Calculate changes within each group
        def calc_group_changes(group):
            group = group.sort_values(time_column)
            group['previous_value'] = group[value_column].shift(1)
            group['yoy_change'] = group[value_column] - group['previous_value']
            group['yoy_change_pct'] = (group['yoy_change'] / group['previous_value'] * 100).replace([np.inf, -np.inf], np.nan)
            return group
        
        result_df = result_df.groupby(group_by).apply(calc_group_changes).reset_index(drop=True)
    else:
        # Calculate changes for entire dataset
        result_df = result_df.sort_values(time_column)
        result_df['previous_value'] = result_df[value_column].shift(1)
        result_df['yoy_change'] = result_df[value_column] - result_df['previous_value']
        result_df['yoy_change_pct'] = (result_df['yoy_change'] / result_df['previous_value'] * 100).replace([np.inf, -np.inf], np.nan)
    
    return result_df

def aggregate_by_dimensions(
    df: pd.DataFrame, 
    dimensions: List[str], 
    value_column: str = 'Value',
    aggregation: str = 'sum'
) -> pd.DataFrame:
    """
    Simple aggregation for analytical brief summaries.
    
    Args:
        df: Dataset DataFrame
        dimensions: List of columns to group by
        value_column: Column to aggregate
        aggregation: Aggregation function ('sum', 'mean', 'count')
        
    Returns:
        Aggregated DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Validate dimensions
    valid_dimensions = [dim for dim in dimensions if dim in df.columns]
    if not valid_dimensions:
        logger.warning("No valid dimensions for aggregation")
        return df
    
    # Validate value column
    if value_column not in df.columns:
        logger.warning(f"Value column {value_column} not found in dataset")
        return df
    
    try:
        if aggregation == 'sum':
            result = df.groupby(valid_dimensions)[value_column].sum().reset_index()
        elif aggregation == 'mean':
            result = df.groupby(valid_dimensions)[value_column].mean().reset_index()
        elif aggregation == 'count':
            result = df.groupby(valid_dimensions)[value_column].count().reset_index()
        else:
            result = df.groupby(valid_dimensions)[value_column].sum().reset_index()
        
        return result
    except Exception as e:
        logger.error(f"Error during aggregation: {str(e)}")
        return df

def get_latest_year_data(df: pd.DataFrame, time_column: str = 'Year') -> pd.DataFrame:
    """
    Get data for the latest available year.
    Useful for current state analysis in briefs.
    
    Args:
        df: Dataset DataFrame
        time_column: Column containing time periods
        
    Returns:
        DataFrame with latest year data only
    """
    if df is None or df.empty or time_column not in df.columns:
        return df
    
    latest_year = df[time_column].max()
    return df[df[time_column] == latest_year].copy()

def calculate_regional_totals(
    df: pd.DataFrame,
    geography_service,
    value_column: str = 'Value',
    latest_year_only: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Calculate totals by geographic classification for analytical briefs.
    
    Args:
        df: Dataset DataFrame
        geography_service: GeographyService instance
        value_column: Column to sum
        latest_year_only: If True, use only latest year data
        
    Returns:
        Dictionary with 'main_regions', 'other_regions', 'countries' DataFrames
    """
    if df is None or df.empty or 'Area' not in df.columns:
        return {'main_regions': pd.DataFrame(), 'other_regions': pd.DataFrame(), 'countries': pd.DataFrame()}
    
    # Use latest year data if requested
    if latest_year_only and 'Year' in df.columns:
        analysis_df = get_latest_year_data(df)
    else:
        analysis_df = df.copy()
    
    # Get geographic classifications
    classification = geography_service.get_area_classification()
    
    results = {}
    
    for geo_type in ['main_regions', 'other_regions', 'countries']:
        areas = classification.get(geo_type, [])
        geo_data = analysis_df[analysis_df['Area'].isin(areas)]
        
        if not geo_data.empty:
            # Sum by area
            totals = geo_data.groupby('Area')[value_column].sum().sort_values(ascending=False).reset_index()
            totals['rank'] = range(1, len(totals) + 1)
            results[geo_type] = totals
        else:
            results[geo_type] = pd.DataFrame()
    
    return results

def filter_for_brief_analysis(
    df: pd.DataFrame,
    items: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    areas: Optional[List[str]] = None,
    year_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Apply filters for analytical brief generation.
    
    Args:
        df: Dataset DataFrame
        items: List of items to include
        elements: List of elements to include  
        areas: List of areas to include
        year_range: Tuple of (start_year, end_year)
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    filtered_df = df.copy()
    
    # Apply filters
    if items and 'Item' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Item'].isin(items)]
    
    if elements and 'Element' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Element'].isin(elements)]
    
    if areas and 'Area' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Area'].isin(areas)]
    
    if year_range and 'Year' in filtered_df.columns:
        start_year, end_year = year_range
        filtered_df = filtered_df[
            (filtered_df['Year'] >= start_year) & 
            (filtered_df['Year'] <= end_year)
        ]
    
    return filtered_df