"""
Data processing utilities for FAOSTAT Analytics Application.

This module provides functions for cleaning, transforming, and analyzing
FAOSTAT data. These utilities support the FAOSTAT service but are separated
to maintain clean separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive cleaning on a FAOSTAT dataset.
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle column names - strip whitespace and standardize
    df.columns = [col.strip() for col in df.columns]
    
    # Convert common columns to appropriate types
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
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to {dtype}: {str(e)}")
    
    # Fill NaN values in numeric columns with appropriate values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Fill NaN values in string columns with empty string
    string_cols = df.select_dtypes(include=['object']).columns
    df[string_cols] = df[string_cols].fillna('')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def detect_anomalies(df: pd.DataFrame, column: str = 'Value', threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalous values in a dataset using Z-score.
    
    Args:
        df: Dataset DataFrame
        column: Column to check for anomalies
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomalies flagged
    """
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()
    
    # Make a copy
    result = df.copy()
    
    # Calculate Z-scores
    mean = result[column].mean()
    std = result[column].std()
    
    if std == 0:  # Avoid division by zero
        result['z_score'] = 0
    else:
        result['z_score'] = (result[column] - mean) / std
    
    # Flag anomalies
    result['is_anomaly'] = abs(result['z_score']) > threshold
    
    return result

def aggregate_by_dimensions(
    df: pd.DataFrame, 
    dimensions: List[str], 
    value_column: str = 'Value',
    aggregation: str = 'sum'
) -> pd.DataFrame:
    """
    Aggregate data by specified dimensions.
    
    Args:
        df: Dataset DataFrame
        dimensions: List of columns to group by
        value_column: Column to aggregate
        aggregation: Aggregation function ('sum', 'mean', 'median', 'min', 'max')
        
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
    
    # Select aggregation function
    agg_funcs = {
        'sum': np.sum,
        'mean': np.mean,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }
    
    agg_func = agg_funcs.get(aggregation.lower(), np.sum)
    
    # Perform aggregation
    try:
        result = df.groupby(valid_dimensions)[value_column].agg(agg_func).reset_index()
        return result
    except Exception as e:
        logger.error(f"Error during aggregation: {str(e)}")
        return df

def calculate_growth_rates(
    df: pd.DataFrame, 
    time_column: str = 'Year', 
    value_column: str = 'Value',
    dimensions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate year-over-year growth rates.
    
    Args:
        df: Dataset DataFrame
        time_column: Column containing time periods
        value_column: Column containing values
        dimensions: Optional list of columns to group by
        
    Returns:
        DataFrame with growth rates
    """
    if df is None or df.empty or time_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()
    
    # Make a copy
    result = df.copy()
    
    # Sort by time column
    result = result.sort_values(time_column)
    
    if dimensions:
        # Calculate growth rates within each group
        valid_dimensions = [dim for dim in dimensions if dim in df.columns]
        
        if valid_dimensions:
            # Group by dimensions and calculate growth rate
            result = result.copy()
            
            # Define a function to calculate growth for each group
            def calculate_group_growth(group):
                group = group.sort_values(time_column)
                group['previous_value'] = group[value_column].shift(1)
                group['growth_rate'] = ((group[value_column] / group['previous_value']) - 1) * 100
                return group
            
            # Apply the function to each group
            result = result.groupby(valid_dimensions).apply(calculate_group_growth).reset_index(drop=True)
    else:
        # Calculate overall growth rate
        result['previous_value'] = result[value_column].shift(1)
        result['growth_rate'] = ((result[value_column] / result['previous_value']) - 1) * 100
    
    return result

def calculate_compound_annual_growth_rate(
    df: pd.DataFrame,
    time_column: str = 'Year',
    value_column: str = 'Value',
    dimensions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate compound annual growth rate (CAGR).
    
    Args:
        df: Dataset DataFrame
        time_column: Column containing time periods
        value_column: Column containing values
        dimensions: Optional list of columns to group by
        
    Returns:
        DataFrame with CAGR calculated
    """
    if df is None or df.empty or time_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()
    
    # Make a copy
    df = df.copy()
    
    # Define CAGR calculation function
    def calculate_cagr(group):
        group = group.sort_values(time_column)
        first_year = group[time_column].iloc[0]
        last_year = group[time_column].iloc[-1]
        first_value = group[value_column].iloc[0]
        last_value = group[value_column].iloc[-1]
        
        # Calculate years difference
        years_diff = last_year - first_year
        
        # Calculate CAGR
        if years_diff > 0 and first_value > 0 and last_value > 0:
            cagr = (((last_value / first_value) ** (1 / years_diff)) - 1) * 100
        else:
            cagr = None
        
        return pd.Series({
            'first_year': first_year,
            'last_year': last_year,
            'first_value': first_value,
            'last_value': last_value,
            'years_diff': years_diff,
            'cagr': cagr
        })
    
    if dimensions:
        # Calculate CAGR for each group
        valid_dimensions = [dim for dim in dimensions if dim in df.columns]
        
        if valid_dimensions:
            result = df.groupby(valid_dimensions).apply(calculate_cagr).reset_index()
            return result
    
    # Calculate overall CAGR
    overall_cagr = calculate_cagr(df)
    return pd.DataFrame([overall_cagr])

def identify_top_bottom_performers(
    df: pd.DataFrame,
    group_by: str,
    value_column: str = 'Value',
    top_n: int = 10,
    bottom_n: int = 10,
    year_filter: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify top and bottom performers in a dataset.
    
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
    
    # Identify top performers
    top_performers = aggregated.nlargest(top_n, value_column)
    
    # Identify bottom performers
    bottom_performers = aggregated.nsmallest(bottom_n, value_column)
    
    return top_performers, bottom_performers

def normalize_time_series(
    df: pd.DataFrame, 
    time_column: str = 'Year', 
    value_column: str = 'Value',
    base_year: Optional[int] = None,
    dimensions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize time series data to a base year (index = 100).
    
    Args:
        df: Dataset DataFrame
        time_column: Column containing time periods
        value_column: Column containing values
        base_year: Year to use as base (100). If None, uses first year.
        dimensions: Optional list of columns to group by
        
    Returns:
        DataFrame with normalized values
    """
    if df is None or df.empty or time_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()
    
    # Make a copy
    result = df.copy()
    
    # Define normalization function
    def normalize_group(group):
        group = group.sort_values(time_column)
        
        # Determine base year value
        if base_year is not None and base_year in group[time_column].values:
            base_value = group.loc[group[time_column] == base_year, value_column].iloc[0]
        else:
            # Use first year as base
            base_value = group[value_column].iloc[0]
        
        # Avoid division by zero
        if base_value == 0:
            group['normalized_value'] = 0
        else:
            group['normalized_value'] = (group[value_column] / base_value) * 100
        
        return group
    
    if dimensions:
        # Normalize within each group
        valid_dimensions = [dim for dim in dimensions if dim in df.columns]
        
        if valid_dimensions:
            result = result.groupby(valid_dimensions).apply(normalize_group).reset_index(drop=True)
            return result
    
    # Normalize overall
    result = normalize_group(result)
    return result

def calculate_correlations(
    df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate correlations between variables.
    
    Args:
        df: Dataset DataFrame
        dimensions: Optional columns to group by before calculating correlations
        variables: List of columns to calculate correlations for
        
    Returns:
        DataFrame with correlation results
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Determine variables to use
    if variables:
        valid_variables = [var for var in variables if var in df.columns]
    else:
        valid_variables = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(valid_variables) < 2:
        logger.warning("Need at least two numeric variables to calculate correlations")
        return pd.DataFrame()
    
    if dimensions:
        # Calculate correlations within each group
        valid_dimensions = [dim for dim in dimensions if dim in df.columns]
        
        if valid_dimensions:
            results = []
            
            for name, group in df.groupby(valid_dimensions):
                # Create single dimension names or handle multiple dimensions
                if isinstance(name, tuple):
                    dim_values = {dim: val for dim, val in zip(valid_dimensions, name)}
                else:
                    dim_values = {valid_dimensions[0]: name}
                
                # Calculate correlation matrix
                corr_matrix = group[valid_variables].corr()
                
                # Extract unique correlations
                for i, var1 in enumerate(valid_variables):
                    for j, var2 in enumerate(valid_variables):
                        if i < j:  # Upper triangle only
                            result = {
                                'variable1': var1,
                                'variable2': var2,
                                'correlation': corr_matrix.loc[var1, var2]
                            }
                            result.update(dim_values)
                            results.append(result)
            
            return pd.DataFrame(results)
    
    # Calculate overall correlations
    corr_matrix = df[valid_variables].corr()
    
    # Extract unique correlations
    results = []
    for i, var1 in enumerate(valid_variables):
        for j, var2 in enumerate(valid_variables):
            if i < j:  # Upper triangle only
                results.append({
                    'variable1': var1,
                    'variable2': var2,
                    'correlation': corr_matrix.loc[var1, var2]
                })
    
    return pd.DataFrame(results)