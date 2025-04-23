"""
FAOSTAT service for interacting with FAO's statistical database APIs.

This module provides a service class for fetching, processing, and analyzing
data from the FAOSTAT API. It includes methods for retrieving dataset metadata,
downloading specific datasets, filtering data, and computing statistics.
"""

import requests
import pandas as pd
import numpy as np
import json
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import lru_cache

from config.constants import DEFAULT_FAOSTAT_URL, DEFAULT_DATASETS_ENDPOINT, DEFAULT_CACHE_DURATION

# Set up logger
logger = logging.getLogger(__name__)

# Define class for FAOSTAT service
class FAOStatService:
    """
    Service for interacting with the FAOSTAT API and datasets.
    
    This class provides methods to:
    - Fetch metadata about available datasets
    - Download specific datasets
    - Filter and process dataset data
    - Calculate summary statistics
    - Cache results for improved performance
    
    Attributes:
        base_url (str): Base URL for the FAOSTAT API
        datasets_endpoint (str): Endpoint for fetching dataset metadata
        cache_duration (int): Duration in seconds to cache API responses
    """
    
    def __init__(
        self, 
        base_url: str = DEFAULT_FAOSTAT_URL,
        datasets_endpoint: str = DEFAULT_DATASETS_ENDPOINT,
        cache_duration: int = DEFAULT_CACHE_DURATION
    ):
        """
        Initialize the FAOSTAT service.
        
        Args:
            base_url: Base URL for the FAOSTAT API
            datasets_endpoint: Endpoint for fetching dataset metadata
            cache_duration: Duration in seconds to cache API responses
        """
        self.base_url = base_url
        self.datasets_endpoint = datasets_endpoint
        self.cache_duration = cache_duration
        
        # Cache for API responses
        self._cache = {}
        self._cache_timestamps = {}
        
        # Lookup tables
        self._dataset_lookup = {}  # For quick dataset code -> metadata lookup
        
        logger.info(f"Initialized FAOSTAT service with base URL: {base_url}")
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """
        Make a request to the FAOSTAT API with error handling.
        
        Args:
            url: Full URL to request
            params: Optional query parameters
            
        Returns:
            requests.Response: The response object
            
        Raises:
            requests.RequestException: If the request fails
        """
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response
        except requests.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get a value from the cache if it exists and is not expired.
        
        Args:
            cache_key: Key to look up in the cache
            
        Returns:
            The cached value or None if not found or expired
        """
        if cache_key in self._cache:
            # Check if cache is expired
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_duration:
                return self._cache[cache_key]
        
        return None
    
    def _set_in_cache(self, cache_key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            cache_key: Key to store the value under
            value: Value to cache
        """
        self._cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.time()
    
    def get_available_datasets(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and return all available FAOSTAT datasets.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame containing dataset information with columns:
            - code: Dataset code
            - name: Dataset name
            - last_updated: Last update date
            - description: Dataset description
            - metadata: Original metadata dictionary
        """
        cache_key = "available_datasets"
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug("Returning cached dataset list")
                return cached_data
        
        try:
            # Construct the URL for the datasets endpoint
            url = f"{self.base_url}{self.datasets_endpoint}"
            
            # Make the request
            logger.info(f"Fetching available datasets from {url}")
            response = self._make_request(url)
            
            # Parse the response
            datasets = response.json()
            
            # Extract useful information
            datasets_info = []
            for dataset in datasets:
                if 'name' in dataset and 'dateUpdate' in dataset:
                    datasets_info.append({
                        'code': dataset.get('code', 'Unknown'),
                        'name': dataset.get('name', 'Unknown'),
                        'last_updated': dataset.get('dateUpdate', 'Unknown'),
                        'description': dataset.get('description', 'No description available'),
                        'metadata': dataset  # Store the full metadata for later use
                    })
            
            # Convert to DataFrame
            df = pd.DataFrame(datasets_info)
            
            # Update the dataset lookup table
            self._update_dataset_lookup(df)
            
            # Cache the result
            self._set_in_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FAOSTAT datasets: {str(e)}")
            # Return empty DataFrame if error occurs
            return pd.DataFrame()
    
    def _update_dataset_lookup(self, datasets_df: pd.DataFrame) -> None:
        """
        Update the internal dataset lookup dictionary.
        
        Args:
            datasets_df: DataFrame containing dataset information
        """
        self._dataset_lookup = {
            row['code']: row['metadata'] for _, row in datasets_df.iterrows()
        }
    
    def get_dataset_metadata(self, dataset_code: str) -> Dict:
        """
        Get metadata for a specific dataset.
        
        Args:
            dataset_code: Code of the dataset
            
        Returns:
            Dictionary containing dataset metadata
        """
        # Check if we already have the lookup table
        if not self._dataset_lookup:
            # Populate the lookup table by getting all datasets
            self.get_available_datasets()
        
        # Return from lookup table if available
        if dataset_code in self._dataset_lookup:
            return self._dataset_lookup[dataset_code]
        
        # If not in lookup table, try to fetch all datasets and check again
        self.get_available_datasets(force_refresh=True)
        
        if dataset_code in self._dataset_lookup:
            return self._dataset_lookup[dataset_code]
        
        # If still not found, return empty dict
        logger.warning(f"Dataset with code '{dataset_code}' not found")
        return {}
    
    def get_dataset(self, dataset_code: str, force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Fetch a specific dataset from FAOSTAT.
        
        Args:
            dataset_code: The code of the dataset to fetch
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Tuple containing:
            - DataFrame of the dataset (or None if error)
            - Dictionary with metadata
        """
        cache_key = f"dataset_{dataset_code}"
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Returning cached dataset {dataset_code}")
                return cached_data
        
        try:
            # Get dataset metadata
            metadata = self.get_dataset_metadata(dataset_code)
            if not metadata:
                return None, {}
            
            # Look for CSV file information
            csv_file_info = None
            for file_info in metadata.get('fileSize', []):
                if file_info.get('format', '').lower() == 'csv':
                    csv_file_info = file_info
                    break
            
            if not csv_file_info:
                logger.error(f"CSV format not available for dataset '{dataset_code}'")
                return None, metadata
            
            # Download the CSV file
            file_url = f"{self.base_url}{csv_file_info.get('id')}"
            logger.info(f"Downloading dataset {dataset_code} from {file_url}")
            
            response = self._make_request(file_url)
            
            # Parse the CSV data
            df = pd.read_csv(BytesIO(response.content), encoding='utf-8')
            
            # Perform basic data cleaning
            df = self._clean_dataset(df)
            
            # Cache the result
            self._set_in_cache(cache_key, (df, metadata))
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error fetching dataset '{dataset_code}': {str(e)}")
            return None, {}
    
    def _clean_dataset(self, df: pd.DataFrame, fill_missing: bool = False) -> pd.DataFrame:
        """
        Perform basic cleaning on a dataset.
        
        Args:
            df: Dataset DataFrame
            fill_missing: Whether to fill missing values (default: False)
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert columns to appropriate types
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # Only fill missing values if explicitly requested
        if fill_missing:
            # Fill NaN values in numeric columns with 0
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Fill NaN values in string columns with empty string
            string_cols = df.select_dtypes(include=['object']).columns
            df[string_cols] = df[string_cols].fillna('')
        
        return df
    
    def get_dataset_preview(self, dataset_code: str, max_rows: int = 10) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Get a preview of the dataset with limited rows.
        
        Args:
            dataset_code: The code of the dataset to preview
            max_rows: Maximum number of rows to return
            
        Returns:
            Tuple containing:
            - Preview DataFrame (or None if error)
            - Dictionary with metadata
        """
        df, metadata = self.get_dataset(dataset_code)
        if df is not None:
            return df.head(max_rows), metadata
        return None, metadata
    
    def filter_dataset(
        self, 
        df: pd.DataFrame, 
        year_range: Optional[Tuple[int, int]] = None,
        regions: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        elements: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter a dataset based on specified criteria.
        
        Args:
            df: The dataset to filter
            year_range: Optional (start_year, end_year) to filter data
            regions: Optional list of regions/countries to include
            items: Optional list of items to include
            elements: Optional list of elements to include
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
            
        filtered_df = df.copy()
        
        # Apply year filter if provided
        if year_range and 'Year' in filtered_df.columns:
            start_year, end_year = year_range
            filtered_df = filtered_df[(filtered_df['Year'] >= start_year) & (filtered_df['Year'] <= end_year)]
        
        # Apply region filter if provided
        if regions and 'Area' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Area'].isin(regions)]
        
        # Apply item filter if provided
        if items and 'Item' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Item'].isin(items)]
        
        # Apply element filter if provided
        if elements and 'Element' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Element'].isin(elements)]
        
        return filtered_df
    
    def get_unique_values(self, df: pd.DataFrame, column: str) -> List:
        """
        Get unique values from a dataset column.
        
        Args:
            df: The dataset
            column: Column name to get unique values from
            
        Returns:
            List of unique values
        """
        if df is None or column not in df.columns:
            return []
        
        return sorted(df[column].unique().tolist())
    
    def get_all_regions(self) -> List[str]:
        """
        Get a comprehensive list of all regions/countries across all datasets.
        
        Returns:
            List of region/country names
        """
        cache_key = "all_regions"
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Get a sample of main datasets that typically have region data
        sample_datasets = ['QCL', 'RFN', 'EF', 'GT']
        
        all_regions = set()
        for dataset_code in sample_datasets:
            try:
                df, _ = self.get_dataset(dataset_code)
                if df is not None and 'Area' in df.columns:
                    all_regions.update(df['Area'].unique())
            except Exception as e:
                logger.warning(f"Error getting regions from dataset {dataset_code}: {str(e)}")
        
        # Convert to sorted list
        regions_list = sorted(list(all_regions))
        
        # Cache the result
        self._set_in_cache(cache_key, regions_list)
        
        return regions_list
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for a dataset.
        
        Args:
            df: The dataset to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None or df.empty:
            return {}
            
        summary = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist()
        }
        
        # Add basic information about common FAOSTAT columns
        if 'Year' in df.columns:
            summary['year_range'] = (int(df['Year'].min()), int(df['Year'].max()))
            summary['year_count'] = int(df['Year'].nunique())
        
        if 'Area' in df.columns:
            summary['area_count'] = int(df['Area'].nunique())
            summary['top_areas'] = df['Area'].value_counts().head(10).to_dict()
        
        if 'Item' in df.columns:
            summary['item_count'] = int(df['Item'].nunique())
            summary['top_items'] = df['Item'].value_counts().head(10).to_dict()
        
        if 'Element' in df.columns:
            summary['element_count'] = int(df['Element'].nunique())
            summary['elements'] = df['Element'].unique().tolist()
        
        if 'Value' in df.columns:
            summary['value_stats'] = {
                'min': float(df['Value'].min()),
                'max': float(df['Value'].max()),
                'mean': float(df['Value'].mean()),
                'median': float(df['Value'].median()),
                'std': float(df['Value'].std())
            }
            
            # Add time series information if available
            if 'Year' in df.columns:
                # Calculate year-over-year change
                yearly_data = df.groupby('Year')['Value'].sum().reset_index()
                yearly_data = yearly_data.sort_values('Year')
                
                if len(yearly_data) > 1:
                    first_year = yearly_data['Year'].iloc[0]
                    last_year = yearly_data['Year'].iloc[-1]
                    first_value = yearly_data['Value'].iloc[0]
                    last_value = yearly_data['Value'].iloc[-1]
                    
                    percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else float('inf')
                    
                    summary['time_series'] = {
                        'first_year': int(first_year),
                        'last_year': int(last_year),
                        'first_value': float(first_value),
                        'last_value': float(last_value),
                        'percent_change': float(percent_change),
                        'direction': 'increase' if percent_change > 0 else 'decrease' if percent_change < 0 else 'unchanged'
                    }
                    
                    # Calculate compound annual growth rate (CAGR)
                    if first_value > 0 and last_value > 0:
                        years_diff = last_year - first_year
                        if years_diff > 0:
                            cagr = (((last_value / first_value) ** (1 / years_diff)) - 1) * 100
                            summary['time_series']['cagr'] = float(cagr)
        
        return summary