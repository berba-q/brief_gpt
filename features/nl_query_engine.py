"""
Natural language query engine for FAOSTAT Analytics Application - FIXED VERSION.

This module provides a service class for processing natural language queries
about FAOSTAT data, converting them into structured analysis parameters,
and executing the appropriate data operations.
It includes intelligent matching capabilities that combine fuzzy matching.
"""

import re
import json
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from rapidfuzz import fuzz, process

from models.data_models import NaturalLanguageQuery, AnalysisFilter
from services.openai_service import OpenAIService
from services.faostat_service import FAOStatService
from utils.prompt_templates import get_structured_query_prompt, get_fact_based_response_prompt

# Set up logger
logger = logging.getLogger(__name__)

class IntelligentFAOSTATMatcher:
    """
    Intelligent matcher for FAOSTAT query terms to dataset values.
    Uses fuzzy matching to work across all domains.
    """
    
    def __init__(self):
        # Disable semantic model to avoid PyTorch issues
        self.semantic_model = None
        logger.info("Semantic model disabled to avoid PyTorch device issues")
        
        # Common term patterns for different query types
        self.semantic_patterns = {
            'production_terms': [
                'production', 'produce', 'producing', 'output', 'harvest', 
                'yield', 'cultivation', 'grown', 'manufacturing'
            ],
            'trade_terms': [
                'trade', 'import', 'export', 'trading', 'shipped', 'sold',
                'buy', 'purchase', 'commerce', 'market'
            ],
            'use_consumption_terms': [
                'use', 'usage', 'consumption', 'consume', 'application',
                'utilization', 'applied', 'employed', 'utilized'
            ],
            'area_land_terms': [
                'area', 'land', 'hectares', 'acres', 'planted', 'cultivated',
                'surface', 'territory', 'space'
            ],
            'price_terms': [
                'price', 'cost', 'value', 'worth', 'rate', 'fee', 'charge'
            ]
        }
    
    def find_best_matches(
        self, 
        query_terms: List[str], 
        available_values: List[str], 
        threshold: float = 70.0,
        max_results: int = 3
    ) -> List[str]:
        """Find best matches using fuzzy matching and patterns."""
        
        if not query_terms or not available_values:
            return []
        
        matched = []
        
        for term in query_terms:
            # Direct fuzzy matching
            fuzzy_matches = process.extract(
                term, 
                available_values, 
                scorer=fuzz.WRatio,
                limit=max_results
            )
            
            # Add matches above threshold
            for match, score in fuzzy_matches:
                if score >= threshold and match not in matched:
                    matched.append(match)
        
        return matched[:max_results]

class NaturalLanguageQueryEngine:
    """
    Enhanced Natural Language Query Engine for FAOSTAT data.
    
    FIXED: Better error handling and working query execution.
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        faostat_service: FAOStatService,
        max_context_history: int = 5
    ):
        """Initialize the natural language query engine."""
        self.openai_service = openai_service
        self.faostat_service = faostat_service
        self.max_context_history = max_context_history
        
        # Initialize intelligent matcher with error handling
        try:
            self.intelligent_matcher = IntelligentFAOSTATMatcher()
            logger.info("Intelligent matcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize intelligent matcher: {e}")
            logger.info("Falling back to basic matching only")
            self.intelligent_matcher = None
        
        # Context and cache
        self.context_history = []
        self.dataset_cache = {}
        self.current_dataset = None
        self.current_dataset_df = None
        
        # Query patterns for different types of questions
        self.query_patterns = self._initialize_query_patterns()
        
        logger.info("Initialized Natural Language Query Engine")
    
    def _initialize_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regex patterns for different query types."""
        return {
            "comparison": {
                "patterns": [
                    r"compare\s+(.+?)\s+(?:between|across|in)\s+(.+)",
                    r"(?:which|what)\s+(.+?)\s+(?:has|have)\s+(?:the\s+)?(?:highest|lowest|most|least)\s+(.+)",
                    r"rank\s+(.+?)\s+by\s+(.+)",
                    r"top\s+(\d+)\s+(.+?)\s+(?:in|for)\s+(.+)"
                ],
                "type": "ranking_comparison"
            },
            "trend": {
                "patterns": [
                    r"trend\s+(?:of|in)\s+(.+?)(?:\s+over\s+time|\s+from\s+(\d{4})\s+to\s+(\d{4}))?",
                    r"(?:how\s+)?(?:has|have)\s+(.+?)\s+changed\s+over\s+time",
                    r"growth\s+(?:rate\s+)?(?:of|in)\s+(.+)",
                    r"increase\s+(?:in|of)\s+(.+)"
                ],
                "type": "time_series"
            },
            "statistical": {
                "patterns": [
                    r"(?:what\s+is\s+the\s+)?(?:average|mean|median|total|sum)\s+(.+?)(?:\s+in\s+(.+))?",
                    r"correlation\s+between\s+(.+?)\s+and\s+(.+)",
                    r"percentage\s+(?:of|share)\s+(.+)"
                ],
                "type": "statistical_analysis"
            }
        }
    
    def process_query(
        self, 
        query: str, 
        dataset_code: Optional[str] = None,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """Process a natural language query and return structured results."""
        
        try:
            # Set dataset if provided
            if dataset_code and dataset_code != self.current_dataset:
                self.set_current_dataset(dataset_code)
            
            if not self.current_dataset or self.current_dataset_df is None:
                return {
                    "error": "No dataset selected. Please select a dataset first.",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add to context history
            if use_context:
                self._add_to_context(query)
            
            # Determine query type and parse
            query_type = self._classify_query(query)
            
            # Use intelligent parsing if available, otherwise fallback to LLM
            if self.intelligent_matcher:
                try:
                    parsed_query = self._parse_query_intelligently(query, query_type)
                except Exception as e:
                    logger.warning(f"Intelligent parsing failed: {e}, falling back to LLM")
                    parsed_query = self._parse_query_with_llm(query, query_type)
            else:
                parsed_query = self._parse_query_with_llm(query, query_type)
            
            # Execute the query
            result = self._execute_query(parsed_query, query)
            
            # Generate response
            response = self._generate_response(query, result, parsed_query)
            
            return {
                "query": query,
                "query_type": query_type,
                "parsed_query": parsed_query,
                "result": result,
                "response": response,
                "dataset_code": self.current_dataset,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            return {
                "error": f"Error processing query: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_query_intelligently(self, query: str, query_type: str) -> Dict[str, Any]:
        """Parse query using intelligent matching instead of just LLM."""
        try:
            # Get dataset context
            dataset_context = self.dataset_cache.get(self.current_dataset, {})
            
            # Extract basic components from query
            components = self._extract_query_components(query)
            
            # Use intelligent matcher to find best matches
            matched_params = {}
            
            # Match years
            if components.get('years'):
                matched_params['year_range'] = (
                    min(components['years']), 
                    max(components['years'])
                )
            
            # Match regions using intelligent matching
            if components.get('regions') and self.intelligent_matcher:
                available_areas = dataset_context.get('unique_area', [])
                matched_regions = self.intelligent_matcher.find_best_matches(
                    components['regions'], 
                    available_areas,
                    threshold=60.0
                )
                if matched_regions:
                    matched_params['regions'] = matched_regions
            
            # Match items using intelligent matching
            if components.get('items') and self.intelligent_matcher:
                available_items = dataset_context.get('unique_item', [])
                matched_items = self.intelligent_matcher.find_best_matches(
                    components['items'], 
                    available_items,
                    threshold=50.0
                )
                if matched_items:
                    matched_params['items'] = matched_items
            
            # Match elements using intelligent matching
            if components.get('elements') and self.intelligent_matcher:
                available_elements = dataset_context.get('unique_element', [])
                matched_elements = self.intelligent_matcher.find_best_matches(
                    components['elements'], 
                    available_elements,
                    threshold=50.0
                )
                if matched_elements:
                    matched_params['elements'] = matched_elements
            
            # Add query type and debug info
            matched_params['query_type'] = query_type
            matched_params['debug_info'] = {
                'extracted_components': components,
                'matching_method': 'intelligent_fuzzy_semantic'
            }
            
            logger.info(f"Intelligent parsing result: {matched_params}")
            return matched_params
            
        except Exception as e:
            logger.error(f"Error in intelligent parsing: {str(e)}")
            # Fallback to original LLM parsing if intelligent parsing fails
            return self._parse_query_with_llm(query, query_type)
    
    def _extract_query_components(self, query: str) -> Dict[str, List[str]]:
        """Extract basic components from natural language query - FIXED year parsing."""
        query_lower = query.lower()
        
        # Extract years - FIXED: Better year pattern matching
        years = []
        # Use findall to get full matches, not groups
        year_matches = re.findall(r'\b(?:19|20)\d{2}\b', query)
        for year_str in year_matches:
            try:
                year = int(year_str)
                if 1961 <= year <= 2025:  # Valid FAOSTAT year range
                    years.append(year)
            except (ValueError, TypeError):
                continue
        
        # Extract potential regions (capitalized words)
        potential_regions = []
        for word in query.split():
            if word.istitle() and len(word) > 3:
                potential_regions.append(word)
        
        # FIXED: Extract potential items (agricultural products/foods)
        # Common agricultural items and food products
        food_items = [
            'wheat', 'rice', 'maize', 'corn', 'barley', 'oats', 'sorghum', 'millet',
            'potato', 'potatoes', 'cassava', 'yam', 'plantain', 'banana', 'bananas',
            'tomato', 'tomatoes', 'onion', 'onions', 'cabbage', 'lettuce', 'carrot', 'carrots',
            'apple', 'apples', 'orange', 'oranges', 'grape', 'grapes', 'lemon', 'lemons',
            'beef', 'pork', 'chicken', 'poultry', 'fish', 'milk', 'cheese', 'eggs',
            'soybean', 'soybeans', 'cotton', 'coffee', 'cocoa', 'tea', 'sugar', 'sugarcane',
            'oil', 'palm', 'sunflower', 'rapeseed', 'olive', 'coconut', 'groundnut',
            'cattle', 'pigs', 'sheep', 'goats', 'buffaloes'
        ]
        
        potential_items = []
        for item in food_items:
            if item in query_lower:
                potential_items.append(item.title())  # Capitalize for matching
        
        # Extract potential elements (measurement types)
        element_indicators = ['production', 'use', 'usage', 'consumption', 'trade', 'import', 'export', 'area', 'yield', 'price']
        potential_elements = []
        for indicator in element_indicators:
            if indicator in query_lower:
                potential_elements.append(indicator)
        
        return {
            'years': years,
            'regions': potential_regions,
            'items': potential_items,
            'elements': potential_elements
        }
    
    def _parse_query_with_llm(self, query: str, query_type: str) -> Dict[str, Any]:
        """FALLBACK: Original LLM parsing method."""
        try:
            dataset_context = self.dataset_cache.get(self.current_dataset, {})
            
            parsed_params = self.openai_service.parse_natural_language_query(
                query, dataset_context, model_type="standard"
            )
            
            if not parsed_params.get("error"):
                parsed_params["query_type"] = query_type
                parsed_params = self._clean_parsed_parameters(parsed_params)
            
            return parsed_params
            
        except Exception as e:
            logger.error(f"Error parsing query with LLM: {str(e)}")
            return {"error": f"Failed to parse query: {str(e)}"}
    
    def _clean_parsed_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameters using intelligent matching if available."""
        dataset_context = self.dataset_cache.get(self.current_dataset, {})
        cleaned = {}
        
        # Extract and validate years
        if "years" in params and params["years"]:
            years = params["years"]
            if isinstance(years, list):
                valid_years = [y for y in years if isinstance(y, int) and 1960 <= y <= 2030]
                if valid_years:
                    cleaned["year_range"] = (min(valid_years), max(valid_years))
        
        # Use intelligent matching for items if available
        if "items" in params and params["items"] and self.intelligent_matcher:
            items = params["items"]
            if isinstance(items, list):
                available_items = dataset_context.get('unique_item', [])
                matched_items = self.intelligent_matcher.find_best_matches(
                    items, available_items, threshold=50.0
                )
                if matched_items:
                    cleaned["items"] = matched_items
                    logger.info(f"Intelligent item matching: {items} → {matched_items}")
        
        # Use intelligent matching for elements if available
        if "elements" in params and params["elements"] and self.intelligent_matcher:
            elements = params["elements"]
            if isinstance(elements, list):
                available_elements = dataset_context.get('unique_element', [])
                matched_elements = self.intelligent_matcher.find_best_matches(
                    elements, available_elements, threshold=50.0
                )
                if matched_elements:
                    cleaned["elements"] = matched_elements
                    logger.info(f"Intelligent element matching: {elements} → {matched_elements}")
        
        # Use intelligent matching for regions if available
        if "regions" in params and params["regions"] and self.intelligent_matcher:
            regions = params["regions"]
            if isinstance(regions, list):
                available_areas = dataset_context.get('unique_area', [])
                matched_regions = self.intelligent_matcher.find_best_matches(
                    regions, available_areas, threshold=60.0
                )
                if matched_regions:
                    cleaned["regions"] = matched_regions
        
        return cleaned
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query based on patterns."""
        query_lower = query.lower()
        
        for category, config in self.query_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    return config["type"]
        
        return "general_query"
    
    def _execute_query(self, parsed_query: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Execute the parsed query on the dataset - FIXED implementation."""
        try:
            # FIXED: Use .empty property instead of boolean evaluation
            if self.current_dataset_df is None or self.current_dataset_df.empty:
                return {"error": "No dataset available for analysis"}
            
            df = self.current_dataset_df.copy()
            
            # Apply filters based on parsed query
            filtered_df = df
            
            # Apply year filter
            if 'year_range' in parsed_query and 'Year' in df.columns:
                start_year, end_year = parsed_query['year_range']
                # Handle the case where both years are the same
                if start_year == end_year:
                    filtered_df = filtered_df[filtered_df['Year'] == start_year]
                else:
                    filtered_df = filtered_df[
                        (filtered_df['Year'] >= start_year) & 
                        (filtered_df['Year'] <= end_year)
                    ]
                logger.info(f"Applied year filter: {start_year}-{end_year}, remaining rows: {len(filtered_df)}")
            
            # Apply regions filter
            if 'regions' in parsed_query and parsed_query['regions'] and 'Area' in df.columns:
                filtered_df = filtered_df[filtered_df['Area'].isin(parsed_query['regions'])]
                logger.info(f"Applied regions filter: {parsed_query['regions']}, remaining rows: {len(filtered_df)}")
            
            # Apply items filter
            if 'items' in parsed_query and parsed_query['items'] and 'Item' in df.columns:
                filtered_df = filtered_df[filtered_df['Item'].isin(parsed_query['items'])]
                logger.info(f"Applied items filter: {parsed_query['items']}, remaining rows: {len(filtered_df)}")
            
            # Apply elements filter
            if 'elements' in parsed_query and parsed_query['elements'] and 'Element' in df.columns:
                filtered_df = filtered_df[filtered_df['Element'].isin(parsed_query['elements'])]
                logger.info(f"Applied elements filter: {parsed_query['elements']}, remaining rows: {len(filtered_df)}")
            
            # FIXED: Check if filtered_df is empty using .empty property
            if filtered_df.empty:
                return {
                    "error": "No data found matching the query criteria",
                    "filtered_rows": 0,
                    "debug_info": parsed_query.get('debug_info', {})
                }
            
            # Route to appropriate query handler based on query type
            query_type = parsed_query.get('query_type', 'general_query')
            
            if query_type in ['ranking_comparison', 'comparison']:
                return self._execute_ranking_query(filtered_df, original_query, parsed_query)
            elif query_type in ['time_series', 'trend']:
                return self._execute_trend_query(filtered_df, original_query, parsed_query)
            elif query_type in ['statistical_analysis', 'statistical']:
                return self._execute_statistical_query(filtered_df, original_query, parsed_query)
            else:
                return self._execute_general_query(filtered_df, original_query, parsed_query)
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"error": f"Query execution failed: {str(e)}"}
    
    def _execute_ranking_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a ranking/comparison query."""
        try:
            if 'Value' not in df.columns:
                return {"error": "No Value column available for ranking analysis"}
            
            # Group by relevant dimensions for ranking
            if 'Area' in df.columns:
                ranking = df.groupby('Area')['Value'].sum().sort_values(ascending=False)
                return {
                    "type": "ranking",
                    "ranking_data": ranking.head(10).to_dict(),
                    "filtered_rows": len(df)
                }
            else:
                return {"error": "No suitable columns found for ranking"}
                
        except Exception as e:
            logger.error(f"Error in ranking query: {e}")
            return {"error": f"Ranking analysis failed: {str(e)}"}
    
    def _execute_trend_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a time series/trend query."""
        try:
            if 'Year' not in df.columns or 'Value' not in df.columns:
                return {"error": "No Year or Value columns available for trend analysis"}
            
            # Calculate trend over time
            trend_data = df.groupby('Year')['Value'].sum().sort_index()
            
            return {
                "type": "trend",
                "trend_data": trend_data.to_dict(),
                "filtered_rows": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in trend query: {e}")
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    def _execute_statistical_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a statistical analysis query."""
        try:
            if 'Value' not in df.columns:
                return {"error": "No Value column available for statistical analysis"}
            
            # Calculate comprehensive statistics
            stats = {
                "count": len(df),
                "mean": float(df['Value'].mean()),
                "median": float(df['Value'].median()),
                "std": float(df['Value'].std()),
                "min": float(df['Value'].min()),
                "max": float(df['Value'].max()),
                "sum": float(df['Value'].sum()),
                "percentiles": {
                    "25th": float(df['Value'].quantile(0.25)),
                    "75th": float(df['Value'].quantile(0.75))
                }
            }
            
            return {
                "type": "statistical",
                "statistics": stats,
                "filtered_rows": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical query: {e}")
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _execute_general_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general data overview query."""
        try:
            # Provide general dataset overview
            overview = {
                "total_records": len(df),
                "columns": list(df.columns)
            }
            
            # Add specific insights based on available columns
            if 'Area' in df.columns:
                overview["countries_covered"] = df['Area'].nunique()
                overview["top_countries"] = df['Area'].value_counts().head(5).to_dict()
            
            if 'Item' in df.columns:
                overview["items_covered"] = df['Item'].nunique()
                overview["top_items"] = df['Item'].value_counts().head(5).to_dict()
            
            if 'Year' in df.columns:
                overview["date_range"] = {
                    "start": int(df['Year'].min()),
                    "end": int(df['Year'].max())
                }
            
            return {
                "type": "factual",
                "data_overview": overview,
                "filtered_rows": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in general query: {e}")
            return {"error": f"General analysis failed: {str(e)}"}
    
    def _generate_response(self, query: str, result: Dict[str, Any], parsed_query: Dict[str, Any]) -> str:
        """Generate a natural language response from query results - FIXED implementation."""
        try:
            if "error" in result:
                return f"I couldn't process your query: {result['error']}"
            
            response_parts = []
            
            # Handle different result types
            result_type = result.get("type", "unknown")
            
            if result_type == "ranking":
                response_parts.append("Here are the ranking results:")
                ranking_data = result.get("ranking_data", {})
                for i, (item, value) in enumerate(list(ranking_data.items())[:5], 1):
                    response_parts.append(f"{i}. {item}: {value:,.0f}")
            
            elif result_type == "trend":
                response_parts.append("Here's the trend analysis:")
                trend_data = result.get("trend_data", {})
                if len(trend_data) > 1:
                    years = sorted(trend_data.keys())
                    start_val = trend_data[years[0]]
                    end_val = trend_data[years[-1]]
                    change = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                    response_parts.append(f"From {years[0]} to {years[-1]}, values changed by {change:.1f}%")
            
            elif result_type == "statistical":
                response_parts.append("Here's the statistical summary:")
                stats = result.get("statistics", {})
                response_parts.append(f"Total records: {stats.get('count', 0):,}")
                response_parts.append(f"Average: {stats.get('mean', 0):,.1f}")
                response_parts.append(f"Range: {stats.get('min', 0):,.0f} to {stats.get('max', 0):,.0f}")
            
            elif result_type == "factual":
                response_parts.append("Here's an overview of the data:")
                overview = result.get("data_overview", {})
                response_parts.append(f"Total records: {overview.get('total_records', 0):,}")
                if 'countries_covered' in overview:
                    response_parts.append(f"Countries covered: {overview['countries_covered']}")
                if 'date_range' in overview:
                    date_range = overview['date_range']
                    response_parts.append(f"Years: {date_range['start']} - {date_range['end']}")
            
            else:
                response_parts.append("Analysis completed successfully.")
                if 'filtered_rows' in result:
                    response_parts.append(f"Found {result['filtered_rows']:,} matching records.")
            
            return " ".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Analysis completed, but I had trouble summarizing the results: {str(e)}"
    
    def set_current_dataset(self, dataset_code: str) -> bool:
        """Set the current dataset for queries."""
        try:
            # Load dataset if not already cached
            if dataset_code not in self.dataset_cache:
                df = self.faostat_service.get_dataset_data(dataset_code)
                if df is None or df.empty:
                    logger.error(f"Failed to load dataset {dataset_code}")
                    return False
                
                # Create dataset context for intelligent matching
                context = {
                    'dataset_code': dataset_code,
                    'unique_area': df['Area'].unique().tolist() if 'Area' in df.columns else [],
                    'unique_item': df['Item'].unique().tolist() if 'Item' in df.columns else [],
                    'unique_element': df['Element'].unique().tolist() if 'Element' in df.columns else [],
                    'year_range': (int(df['Year'].min()), int(df['Year'].max())) if 'Year' in df.columns else None,
                    'total_records': len(df)
                }
                
                self.dataset_cache[dataset_code] = context
                self.current_dataset_df = df
            
            self.current_dataset = dataset_code
            logger.info(f"Set current dataset to {dataset_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting dataset {dataset_code}: {str(e)}")
            return False
    
    def _add_to_context(self, query: str) -> None:
        """Add query to context history."""
        timestamp = datetime.now().isoformat()
        self.context_history.append({
            "query": query,
            "timestamp": timestamp
        })
        
        # Keep only recent context
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def get_context_summary(self) -> str:
        """Get a summary of recent queries for context."""
        if not self.context_history:
            return "No recent queries."
        
        recent_queries = [item["query"] for item in self.context_history[-3:]]
        return f"Recent queries: {'; '.join(recent_queries)}"
    
    def clear_context(self) -> None:
        """Clear the context history."""
        self.context_history = []
        logger.info("Cleared context history")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        if not self.current_dataset:
            return {"error": "No dataset selected"}
        
        context = self.dataset_cache.get(self.current_dataset, {})
        return {
            "dataset_code": self.current_dataset,
            "context": context,
            "has_data": self.current_dataset_df is not None and not self.current_dataset_df.empty
        }