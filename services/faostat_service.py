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
            response_data = response.json()
            
            # FIXED: Handle the nested structure properly (based on debugger findings)
            datasets = None
            
            if isinstance(response_data, dict) and "Datasets" in response_data:
                datasets_container = response_data["Datasets"]
                logger.info(f"Found Datasets container: {type(datasets_container)}")
                
                if isinstance(datasets_container, dict):
                    if "Dataset" in datasets_container:
                        # Standard case: datasets at ["Datasets"]["Dataset"]
                        datasets = datasets_container["Dataset"]
                        logger.info(f"Found datasets at ['Datasets']['Dataset']: {type(datasets)}, length: {len(datasets) if isinstance(datasets, list) else 'single'}")
                    else:
                        logger.warning("No 'Dataset' key found in Datasets container")
                        logger.info(f"Available keys: {list(datasets_container.keys())}")
                        return pd.DataFrame()
                elif isinstance(datasets_container, list):
                    # Direct list case
                    datasets = datasets_container
                    logger.info(f"Datasets container is direct list: {len(datasets)} items")
            else:
                logger.error("No 'Datasets' key found in response")
                logger.info(f"Available keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
                return pd.DataFrame()
            
            if datasets is None:
                logger.error("Could not find datasets in response")
                return pd.DataFrame()
            
            # Normalize to list format
            if isinstance(datasets, dict):
                # Single dataset - convert to list
                logger.info("Single dataset found, converting to list")
                datasets = [datasets]
            elif isinstance(datasets, list):
                # Multiple datasets - use as is
                logger.info(f"Multiple datasets found: {len(datasets)} items")
            else:
                logger.error(f"Unexpected datasets type: {type(datasets)}")
                return pd.DataFrame()
            
            # Extract useful information using correct field names
            datasets_info = []
            for i, dataset in enumerate(datasets):
                try:
                    if isinstance(dataset, dict):
                        # Use the correct field names that FAOSTAT actually uses
                        name = (dataset.get('DatasetName') or 
                            dataset.get('name') or 
                            dataset.get('Name') or 
                            'Unknown')
                        
                        code = (dataset.get('DatasetCode') or 
                            dataset.get('code') or 
                            dataset.get('Code') or 
                            f"dataset_{i}")
                        
                        date_updated = (dataset.get('DateUpdate') or 
                                    dataset.get('dateUpdate') or 
                                    dataset.get('last_updated') or 
                                    'Unknown')
                        
                        description = (dataset.get('DatasetDescription') or 
                                    dataset.get('description') or 
                                    dataset.get('Topic') or
                                    'No description available')
                        
                        # Only add if we have meaningful data
                        if name != 'Unknown' and code != f"dataset_{i}":
                            datasets_info.append({
                                'code': code,
                                'name': name,
                                'last_updated': date_updated,
                                'description': description,
                                'metadata': dataset  # Store the full metadata for later use
                            })
                            logger.debug(f"Successfully processed dataset: {name} ({code})")
                        else:
                            logger.warning(f"Skipping dataset {i} with insufficient data")
                            logger.debug(f"Dataset {i} keys: {list(dataset.keys())}")
                            
                except Exception as dataset_error:
                    logger.error(f"Error processing dataset {i}: {str(dataset_error)}")
                    continue
            
            # Convert to DataFrame
            df = pd.DataFrame(datasets_info)
            
            # Log the results
            logger.info(f"Successfully processed {len(datasets_info)} out of {len(datasets)} datasets")
            if len(datasets_info) > 0:
                logger.info(f"Sample dataset names: {[d['name'] for d in datasets_info[:3]]}")
            else:
                logger.error("No datasets were successfully processed!")
                # Log a sample for debugging
                if len(datasets) > 0:
                    sample_dataset = datasets[0]
                    logger.error(f"Sample dataset keys: {list(sample_dataset.keys()) if isinstance(sample_dataset, dict) else type(sample_dataset)}")
            
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
            # FIXED: Get dataset info directly from the lookup (no separate metadata call)
            if not self._dataset_lookup:
                self.get_available_datasets()
            
            if dataset_code not in self._dataset_lookup:
                logger.error(f"Dataset '{dataset_code}' not found in available datasets")
                return None, {}
            
            dataset_info = self._dataset_lookup[dataset_code]
            logger.info(f"Dataset info keys: {list(dataset_info.keys())}")
            
            # FIXED: Look for fileLocation directly in the dataset JSON response
            download_url = None
            is_zip_file = False
            
            # Strategy 1: Look for fileLocation field (primary FAOSTAT pattern)
            if 'fileLocation' in dataset_info and dataset_info['fileLocation']:
                download_url = dataset_info['fileLocation']
                # Ensure it's a full URL
                if not download_url.startswith('http'):
                    download_url = f"{self.base_url.rstrip('/')}/{download_url.lstrip('/')}"
                
                # FAOSTAT typically serves ZIP files
                is_zip_file = download_url.endswith('.zip')
                
                logger.info(f"Found fileLocation: {download_url} (ZIP: {is_zip_file})")
            
            # Strategy 2: Look for other common FAOSTAT fields in the JSON response
            elif any(field in dataset_info for field in ['downloadUrl', 'DownloadUrl', 'url', 'FileLocation']):
                for field in ['downloadUrl', 'DownloadUrl', 'url', 'FileLocation']:
                    if field in dataset_info and dataset_info[field]:
                        download_url = dataset_info[field]
                        if not download_url.startswith('http'):
                            download_url = f"{self.base_url.rstrip('/')}/{download_url.lstrip('/')}"
                        is_zip_file = download_url.endswith('.zip')
                        logger.info(f"Found download URL in field '{field}': {download_url}")
                        break
            
            # Strategy 3: Construct URL based on dataset code (FAOSTAT bulk download pattern)
            else:
                # FAOSTAT bulk download patterns
                possible_patterns = [
                    f"{self.base_url.rstrip('/')}/{dataset_code}.zip",
                    f"{self.base_url.rstrip('/')}/{dataset_code}_E.zip",  # English version
                    f"{self.base_url.rstrip('/')}/bulk/{dataset_code}.zip"
                ]
                
                for pattern_url in possible_patterns:
                    try:
                        # Test if this URL exists (HEAD request)
                        test_response = requests.head(pattern_url, timeout=10)
                        if test_response.status_code == 200:
                            download_url = pattern_url
                            is_zip_file = True
                            logger.info(f"Found working ZIP pattern: {download_url}")
                            break
                    except:
                        continue
            
            if not download_url:
                logger.error(f"Could not find download URL for dataset '{dataset_code}'")
                logger.error(f"Available dataset fields: {list(dataset_info.keys())}")
                return None, dataset_info
            
            # Download and process the file
            logger.info(f"Downloading dataset {dataset_code} from {download_url}")
            response = self._make_request(download_url)
            
            # Handle ZIP files (FAOSTAT standard)
            if is_zip_file:
                logger.info("Processing ZIP file...")
                
                import zipfile
                from io import BytesIO
                
                try:
                    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                        # List files in the ZIP
                        file_list = zip_file.namelist()
                        logger.info(f"Files in ZIP: {file_list}")
                        
                        # Look for CSV file in the ZIP
                        csv_files = [f for f in file_list if f.endswith('.csv')]
                        
                        if not csv_files:
                            logger.error(f"No CSV files found in ZIP for dataset {dataset_code}")
                            return None, dataset_info
                        
                        # Use the first CSV file (or the one that matches the dataset code)
                        csv_filename = csv_files[0]
                        for f in csv_files:
                            if dataset_code.lower() in f.lower():
                                csv_filename = f
                                break
                        
                        logger.info(f"Extracting CSV file: {csv_filename}")
                        
                        # Extract and read the CSV
                        with zip_file.open(csv_filename) as csv_file:
                            csv_content = csv_file.read()
                            
                            # Try different encodings
                            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                                try:
                                    df = pd.read_csv(BytesIO(csv_content), encoding=encoding)
                                    logger.info(f"Successfully parsed CSV with {encoding} encoding")
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                logger.error("Could not parse CSV with any common encoding")
                                return None, dataset_info
                                
                except zipfile.BadZipFile:
                    logger.error(f"Downloaded file is not a valid ZIP file")
                    return None, dataset_info
                    
            else:
                # Handle direct CSV files
                logger.info("Processing direct CSV file...")
                
                # Check if response is actually CSV
                content_type = response.headers.get('content-type', '').lower()
                if 'text/csv' not in content_type and 'application/csv' not in content_type:
                    # Try to detect CSV by content
                    content_sample = response.content[:1000].decode('utf-8', errors='ignore')
                    if not (',' in content_sample or ';' in content_sample or '\t' in content_sample):
                        logger.error(f"Response doesn't appear to be CSV. Content-Type: {content_type}")
                        return None, dataset_info
                
                # Parse the CSV data with multiple encoding attempts
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(BytesIO(response.content), encoding=encoding)
                        logger.info(f"Successfully parsed CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.error("Could not parse CSV with any common encoding")
                    return None, dataset_info
            
            if df.empty:
                logger.warning(f"Dataset {dataset_code} is empty")
                return df, dataset_info
            
            logger.info(f"Successfully loaded dataset {dataset_code}: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Perform basic data cleaning
            df = self._clean_dataset(df)
            
            # Cache the result
            self._set_in_cache(cache_key, (df, dataset_info))
            
            return df, dataset_info
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching dataset '{dataset_code}': {str(e)}")
            return None, {}
        except Exception as e:
            logger.error(f"Error fetching dataset '{dataset_code}': {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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