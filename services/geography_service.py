"""
Geography Service for FAOSTAT area code classification.

This service handles the classification of FAOSTAT area codes into:
- Main regions (World, Africa, Asia, Americas, Europe, Oceania) - area codes > 1000
- Other regions - area codes > 1000, excluding main regions  
- Countries - area codes < 1000
"""

import logging
from typing import List, Dict, Set, Optional
import pandas as pd

from services.faostat_service import FAOStatService

logger = logging.getLogger(__name__)

class GeographyService:
    """Service for handling FAOSTAT geographic classifications."""
    
    def __init__(self, faostat_service: FAOStatService):
        """
        Initialize geography service.
        
        Args:
            faostat_service: Instance of FAOStatService for data access
        """
        self.faostat_service = faostat_service
        self._area_cache = {}
        
        # Define main regions (these are the primary FAO regional groupings)
        self.main_regions = {
            "World",
            "Africa", 
            "Asia",
            "Americas",
            "Europe",
            "Oceania"
        }
    
    def get_area_classification(self) -> Dict[str, List[str]]:
        """
        Get classification of all areas into main regions, other regions, and countries.
        
        Returns:
            Dictionary with keys 'main_regions', 'other_regions', 'countries'
        """
        cache_key = "area_classification"
        
        if cache_key in self._area_cache:
            return self._area_cache[cache_key]
        
        try:
            # Get area data from a representative dataset
            # Use RFN (Fertilizer by nutrient) as it has comprehensive area coverage
            df, _ = self.faostat_service.get_dataset('RFN')
            
            if df is None or 'Area' not in df.columns:
                logger.warning("Could not load area data for classification")
                return self._get_fallback_classification()
            
            # Get unique areas
            all_areas = set(df['Area'].unique())
            
            # If we have area codes, use them for classification
            if 'Area Code' in df.columns:
                area_codes = df[['Area', 'Area Code']].drop_duplicates()
                return self._classify_by_area_codes(area_codes, all_areas)
            else:
                # Fall back to name-based classification
                return self._classify_by_names(all_areas)
                
        except Exception as e:
            logger.error(f"Error classifying areas: {str(e)}")
            return self._get_fallback_classification()
    
    def _classify_by_area_codes(self, area_codes_df: pd.DataFrame, all_areas: Set[str]) -> Dict[str, List[str]]:
        """Classify areas using area codes (preferred method)."""
        
        main_regions = []
        other_regions = []
        countries = []
        
        for _, row in area_codes_df.iterrows():
            area_name = row['Area']
            area_code = row['Area Code']
            
            try:
                code_num = int(area_code)
                
                if code_num < 1000:
                    # This is a country (area codes < 1000)
                    countries.append(area_name)
                else:
                    # This is a regional grouping (area codes >= 1000)
                    if area_name in self.main_regions:
                        main_regions.append(area_name)
                    else:
                        other_regions.append(area_name)
                        
            except (ValueError, TypeError):
                # Handle non-numeric codes - classify by name
                if area_name in self.main_regions:
                    main_regions.append(area_name)
                else:
                    # If we can't determine from area code, assume it's a region
                    other_regions.append(area_name)
        
        # Cache and return
        classification = {
            'main_regions': sorted(main_regions),
            'other_regions': sorted(other_regions), 
            'countries': sorted(countries)
        }
        
        self._area_cache["area_classification"] = classification
        return classification
    
    def _classify_by_names(self, all_areas: Set[str]) -> Dict[str, List[str]]:
        """Fallback classification using area names only."""
        
        main_regions = []
        other_regions = []
        countries = []
        
        for area in all_areas:
            if area in self.main_regions:
                main_regions.append(area)
            else:
                # Without area codes, we can't reliably distinguish countries from other regions
                # So we'll put everything else in other_regions as a safe fallback
                other_regions.append(area)
        
        # Cache and return
        classification = {
            'main_regions': sorted(main_regions),
            'other_regions': sorted(other_regions),
            'countries': sorted(countries)  # Will be empty in fallback mode
        }
        
        self._area_cache["area_classification"] = classification
        return classification
    
    def _get_fallback_classification(self) -> Dict[str, List[str]]:
        """Provide fallback classification if data loading fails."""
        return {
            'main_regions': list(self.main_regions),
            'other_regions': [],
            'countries': []
        }
    
    def get_main_regions(self) -> List[str]:
        """Get list of main regions."""
        classification = self.get_area_classification()
        return classification['main_regions']
    
    def get_other_regions(self) -> List[str]:
        """Get list of other regions."""
        classification = self.get_area_classification()
        return classification['other_regions']
    
    def get_countries(self) -> List[str]:
        """Get list of countries."""
        classification = self.get_area_classification()
        return classification['countries']
    
    def get_top_countries_by_metric(
        self, 
        df: pd.DataFrame, 
        metric: str = 'Value',
        n: int = 10,
        ascending: bool = False
    ) -> List[str]:
        """
        Get top N countries by a specific metric.
        
        Args:
            df: Dataset DataFrame
            metric: Column name to rank by
            n: Number of countries to return
            ascending: If True, get bottom N instead of top N
            
        Returns:
            List of country names
        """
        if df is None or df.empty or 'Area' not in df.columns or metric not in df.columns:
            return []
        
        # Filter to countries only
        countries = self.get_countries()
        country_data = df[df['Area'].isin(countries)]
        
        if country_data.empty:
            return []
        
        # Calculate totals by country
        country_totals = country_data.groupby('Area')[metric].sum()
        
        # Get top/bottom N
        if ascending:
            top_countries = country_totals.nsmallest(n)
        else:
            top_countries = country_totals.nlargest(n)
        
        return top_countries.index.tolist()
    
    def classify_area(self, area_name: str) -> str:
        """
        Classify a single area into main_region, other_region, or country.
        
        Args:
            area_name: Name of the area to classify
            
        Returns:
            Classification: 'main_region', 'other_region', or 'country'
        """
        classification = self.get_area_classification()
        
        if area_name in classification['main_regions']:
            return 'main_region'
        elif area_name in classification['other_regions']:
            return 'other_region'
        elif area_name in classification['countries']:
            return 'country'
        else:
            return 'unknown'
    
    def get_areas_by_type(self, area_type: str) -> List[str]:
        """
        Get areas of a specific type.
        
        Args:
            area_type: 'main_regions', 'other_regions', or 'countries'
            
        Returns:
            List of area names
        """
        classification = self.get_area_classification()
        return classification.get(area_type, [])
    
    def filter_areas_by_availability(self, df: pd.DataFrame, area_type: str) -> List[str]:
        """
        Filter areas by type and availability in the dataset.
        
        Args:
            df: Dataset DataFrame
            area_type: 'main_regions', 'other_regions', or 'countries'
            
        Returns:
            List of available areas of the specified type
        """
        if df is None or 'Area' not in df.columns:
            return []
        
        available_areas = set(df['Area'].unique())
        type_areas = self.get_areas_by_type(area_type)
        
        return [area for area in type_areas if area in available_areas]