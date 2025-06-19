"""
Natural language query engine for FAOSTAT Analytics Application - FIXED VERSION.

This module provides a service class for processing natural language queries
about FAOSTAT data, converting them into structured analysis parameters,
and executing the appropriate data operations.
It includes intelligent matching capabilities that combine fuzzy matching,
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
        
        if not available_values:
            return []
        
        all_matches = {}
        
        # Process each query term
        for query_term in query_terms:
            if not query_term or not query_term.strip():
                continue
                
            query_term = query_term.strip()
            
            # Get fuzzy matches
            try:
                fuzzy_matches = self._fuzzy_match(query_term, available_values, threshold * 0.8)
            except Exception as e:
                logger.warning(f"Fuzzy matching failed for '{query_term}': {e}")
                fuzzy_matches = []
            
            # Get pattern matches
            try:
                pattern_matches = self._pattern_match(query_term, available_values)
            except Exception as e:
                logger.warning(f"Pattern matching failed for '{query_term}': {e}")
                pattern_matches = []
            
            # Combine matches for this term
            term_matches = self._combine_matches(fuzzy_matches, pattern_matches, [])
            
            # Add to overall matches with query term as key
            for match in term_matches[:max_results]:
                if match not in all_matches:
                    all_matches[match] = []
                all_matches[match].append(query_term)
        
        # Return matches sorted by number of matching terms
        if not all_matches:
            return []
        
        sorted_matches = sorted(all_matches.items(), key=lambda x: len(x[1]), reverse=True)
        return [match for match, _ in sorted_matches[:max_results]]
    
    def _fuzzy_match(self, query_term: str, available_values: List[str], threshold: float = 70.0) -> List[Tuple[str, float]]:
        """Fuzzy string matching using rapidfuzz."""
        if not available_values:
            return []
            
        try:
            matches = process.extract(
                query_term,
                available_values,
                scorer=fuzz.WRatio,
                limit=5
            )
            return [(match[0], match[1]) for match in matches if match[1] >= threshold]
        except Exception as e:
            logger.error(f"Fuzzy matching error: {e}")
            return []
    
    def _pattern_match(self, query_term: str, available_values: List[str]) -> List[Tuple[str, float]]:
        """Pattern-based matching using semantic categories."""
        query_lower = query_term.lower()
        matches = []
        
        # Find which semantic category the query term belongs to
        query_category = None
        for category, terms in self.semantic_patterns.items():
            if any(term in query_lower or query_lower in term for term in terms):
                query_category = category
                break
        
        if not query_category:
            return []
        
        # Score available values based on category keywords
        category_keywords = self.semantic_patterns[query_category]
        
        for value in available_values:
            value_lower = value.lower()
            score = 0
            
            # Check how many category keywords appear in the value
            for keyword in category_keywords:
                if keyword in value_lower:
                    score += 80
                elif any(kw in value_lower for kw in [keyword[:4], keyword[:3]]):
                    score += 60
            
            # Bonus for query term appearing in value
            if query_lower in value_lower:
                score += 90
            elif any(word in value_lower for word in query_lower.split()):
                score += 70
            
            if score > 50:
                matches.append((value, min(score, 100)))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def _combine_matches(self, fuzzy_matches, pattern_matches, semantic_matches) -> List[str]:
        """Combine and rank matches from different methods."""
        scores = {}
        
        # Add fuzzy match scores (weight: 1.0)
        for value, score in fuzzy_matches:
            scores[value] = scores.get(value, 0) + score * 1.0
        
        # Add pattern match scores (weight: 0.8)
        for value, score in pattern_matches:
            scores[value] = scores.get(value, 0) + score * 0.8
        
        # Sort by combined score
        ranked_values = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [value for value, score in ranked_values]


class NaturalLanguageQueryEngine:
    """
    Engine for processing natural language queries about FAOSTAT data.
    
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
            if not self.current_dataset_df or self.current_dataset_df.empty:
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
            
            if filtered_df.empty:
                return {
                    "error": "No data found matching the query criteria",
                    "filtered_rows": 0,
                    "total_rows": len(df)
                }
            
            # Determine query type and generate appropriate analysis
            query_type = parsed_query.get('query_type', 'general_query')
            
            if query_type == 'ranking_comparison':
                return self._execute_ranking_query(filtered_df, original_query, parsed_query)
            elif query_type == 'time_series':
                return self._execute_trend_query(filtered_df, original_query, parsed_query)
            elif query_type == 'statistical_analysis':
                return self._execute_statistical_query(filtered_df, original_query, parsed_query)
            else:
                return self._execute_general_query(filtered_df, original_query, parsed_query)
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {"error": f"Query execution failed: {str(e)}"}
    
    def _execute_ranking_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a ranking/comparison query - FIXED implementation."""
        try:
            if 'Value' not in df.columns:
                return {"error": "No Value column available for ranking"}
            
            # Group by Area (countries) and sum values
            if 'Area' in df.columns:
                country_totals = df.groupby('Area')['Value'].sum().reset_index()
                country_totals = country_totals.sort_values('Value', ascending=False)
                
                # Get top performers (limit to prevent too much data)
                top_countries = country_totals.head(10)
                
                # Calculate statistics
                stats = {
                    'total_countries': len(country_totals),
                    'global_total': float(country_totals['Value'].sum()),
                    'average': float(country_totals['Value'].mean()),
                    'median': float(country_totals['Value'].median())
                }
                
                return {
                    "type": "ranking",
                    "top_10": dict(zip(top_countries['Area'], top_countries['Value'])),
                    "data_summary": {
                        "total_entities": len(country_totals),
                        "highest": {
                            "name": top_countries.iloc[0]['Area'],
                            "value": float(top_countries.iloc[0]['Value'])
                        },
                        "average": stats['average']
                    },
                    "filtered_rows": len(df),
                    "query_info": parsed_query
                }
            else:
                return {"error": "No Area column available for country ranking"}
                
        except Exception as e:
            logger.error(f"Error in ranking query: {e}")
            return {"error": f"Ranking analysis failed: {str(e)}"}
    
    def _execute_trend_query(self, df: pd.DataFrame, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trend/time series query."""
        try:
            if 'Year' not in df.columns or 'Value' not in df.columns:
                return {"error": "No Year or Value columns available for trend analysis"}
            
            # Group by year and sum values
            yearly_data = df.groupby('Year')['Value'].sum().reset_index()
            yearly_data = yearly_data.sort_values('Year')
            
            if len(yearly_data) < 2:
                return {"error": "Need at least 2 years of data for trend analysis"}
            
            # Calculate trend statistics
            first_year = yearly_data.iloc[0]
            last_year = yearly_data.iloc[-1]
            
            total_change = last_year['Value'] - first_year['Value']
            percent_change = (total_change / first_year['Value']) * 100 if first_year['Value'] != 0 else 0
            
            # Calculate CAGR if multiple years
            years_diff = last_year['Year'] - first_year['Year']
            cagr = 0
            if years_diff > 0 and first_year['Value'] > 0:
                cagr = (((last_year['Value'] / first_year['Value']) ** (1 / years_diff)) - 1) * 100
            
            return {
                "type": "trend",
                "time_series": yearly_data.set_index('Year')['Value'].to_dict(),
                "trend_summary": {
                    "first_year": int(first_year['Year']),
                    "last_year": int(last_year['Year']),
                    "first_value": float(first_year['Value']),
                    "last_value": float(last_year['Value']),
                    "total_change_percent": float(percent_change),
                    "cagr_percent": float(cagr)
                },
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
            
            result_type = result.get("type", "unknown")
            
            if result_type == "ranking":
                return self._generate_ranking_response(query, result)
            elif result_type == "trend":
                return self._generate_trend_response(query, result)
            elif result_type == "statistical":
                return self._generate_statistical_response(query, result)
            elif result_type == "factual":
                return self._generate_factual_response(query, result)
            else:
                return "I processed your query but couldn't generate a specific response format."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating the response."
    
    def _generate_ranking_response(self, query: str, result: Dict[str, Any]) -> str:
        """Generate response for ranking queries - FIXED formatting."""
        try:
            top_10 = result.get("top_10", {})
            data_summary = result.get("data_summary", {})
            
            if not top_10:
                return "No ranking data was found for your query."
            
            # Get top 5 for response
            top_5_items = list(top_10.items())[:5]
            
            response_parts = []
            
            # Add context about the query
            if "nitrogen" in query.lower() and "2022" in query:
                response_parts.append("Based on the nitrogen fertilizer data for 2022, here are the top users:")
            elif "fertilizer" in query.lower():
                response_parts.append("Based on the fertilizer data, here are the top countries:")
            else:
                response_parts.append("Based on your query, here are the top performers:")
            
            # List top 5
            response_parts.append("\n**Top 5:**")
            for i, (country, value) in enumerate(top_5_items, 1):
                # Format large numbers nicely
                if value >= 1_000_000:
                    formatted_value = f"{value/1_000_000:.1f}M"
                elif value >= 1_000:
                    formatted_value = f"{value/1_000:.1f}K" 
                else:
                    formatted_value = f"{value:.1f}"
                response_parts.append(f"{i}. **{country}**: {formatted_value}")
            
            # Add summary statistics
            if data_summary:
                highest = data_summary.get("highest", {})
                if highest:
                    highest_val = highest.get('value', 0)
                    if highest_val >= 1_000_000:
                        formatted_highest = f"{highest_val/1_000_000:.1f}M"
                    elif highest_val >= 1_000:
                        formatted_highest = f"{highest_val/1_000:.1f}K"
                    else:
                        formatted_highest = f"{highest_val:.1f}"
                    response_parts.append(f"\n📊 **Leading country**: {highest.get('name')} with {formatted_highest}")
                
                total_entities = data_summary.get("total_entities", 0)
                if total_entities > 0:
                    response_parts.append(f"🌍 **Countries analyzed**: {total_entities}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating ranking response: {e}")
            return "Found ranking data but couldn't format the response properly."
    
    def _generate_trend_response(self, query: str, result: Dict[str, Any]) -> str:
        """Generate response for trend queries."""
        try:
            trend_summary = result.get("trend_summary", {})
            
            if not trend_summary:
                return "No trend data was found for your query."
            
            response_parts = []
            response_parts.append("Here's the trend analysis for your query:")
            
            # Time period
            first_year = trend_summary.get("first_year")
            last_year = trend_summary.get("last_year")
            if first_year and last_year:
                response_parts.append(f"\n📅 **Period**: {first_year} - {last_year}")
            
            # Overall change
            total_change = trend_summary.get("total_change_percent", 0)
            if total_change != 0:
                direction = "increased" if total_change > 0 else "decreased"
                response_parts.append(f"📈 **Overall change**: {direction} by {abs(total_change):.1f}%")
            
            # Growth rates
            cagr = trend_summary.get("cagr_percent", 0)
            if abs(cagr) > 0.1:
                response_parts.append(f"📊 **Compound Annual Growth Rate**: {cagr:+.1f}%")
            
            # Values
            first_value = trend_summary.get("first_value", 0)
            last_value = trend_summary.get("last_value", 0)
            if first_value and last_value:
                response_parts.append(f"🔢 **Values**: {first_value:,.0f} → {last_value:,.0f}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating trend response: {e}")
            return "Found trend data but couldn't format the response properly."
    
    def _generate_statistical_response(self, query: str, result: Dict[str, Any]) -> str:
        """Generate response for statistical queries."""
        try:
            stats = result.get("statistics", {})
            
            if not stats:
                return "No statistical data was found for your query."
            
            response_parts = []
            response_parts.append("Here are the statistical insights for your query:")
            
            # Basic statistics
            count = stats.get("count", 0)
            mean_val = stats.get("mean", 0)
            median_val = stats.get("median", 0)
            
            response_parts.append(f"\n📊 **Dataset size**: {count:,} records")
            
            if mean_val > 0:
                response_parts.append(f"📈 **Average value**: {mean_val:,.1f}")
            
            if median_val > 0:
                response_parts.append(f"📊 **Median value**: {median_val:,.1f}")
            
            # Range
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            if min_val != max_val:
                response_parts.append(f"📏 **Range**: {min_val:,.1f} - {max_val:,.1f}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating statistical response: {e}")
            return "Found statistical data but couldn't format the response properly."
    
    def _generate_factual_response(self, query: str, result: Dict[str, Any]) -> str:
        """Generate response for factual/overview queries."""
        try:
            overview = result.get("data_overview", {})
            
            if not overview:
                return "No overview data was found for your query."
            
            response_parts = []
            response_parts.append("Here's what I found in the data:")
            
            # Basic dataset info
            total_records = overview.get("total_records", 0)
            response_parts.append(f"\n📊 **Total records**: {total_records:,}")
            
            # Geographic coverage
            countries_covered = overview.get("countries_covered")
            if countries_covered:
                response_parts.append(f"🌍 **Countries covered**: {countries_covered}")
            
            # Items coverage
            items_covered = overview.get("items_covered")
            if items_covered:
                response_parts.append(f"📦 **Items/commodities**: {items_covered}")
            
            # Date range
            date_range = overview.get("date_range")
            if date_range:
                start = date_range.get("start")
                end = date_range.get("end")
                response_parts.append(f"📅 **Time period**: {start} - {end}")
            
            # Top countries
            top_countries = overview.get("top_countries")
            if top_countries:
                top_3 = list(top_countries.items())[:3]
                response_parts.append(f"🏆 **Most represented countries**: {', '.join([country for country, _ in top_3])}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating factual response: {e}")
            return "Found overview data but couldn't format the response properly."
    
    def set_current_dataset(self, dataset_code: str, dataset_df: Optional[pd.DataFrame] = None):
        """Set the current dataset for queries."""
        self.current_dataset = dataset_code
        self.current_dataset_df = dataset_df
        
        # Cache dataset metadata if not already cached
        if dataset_code not in self.dataset_cache and dataset_df is not None:
            try:
                self.dataset_cache[dataset_code] = {
                    'unique_area': dataset_df['Area'].unique().tolist() if 'Area' in dataset_df.columns else [],
                    'unique_item': dataset_df['Item'].unique().tolist() if 'Item' in dataset_df.columns else [],
                    'unique_element': dataset_df['Element'].unique().tolist() if 'Element' in dataset_df.columns else [],
                    'columns': dataset_df.columns.tolist(),
                    'shape': dataset_df.shape
                }
            except Exception as e:
                logger.warning(f"Could not cache dataset metadata: {e}")
    
    def _add_to_context(self, query: str):
        """Add query to context history."""
        self.context_history.append({
            'query': query,
            'timestamp': datetime.now()
        })
        
        # Keep only recent context
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def clear_context(self):
        """Clear the context history."""
        self.context_history = []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        return {
            'current_dataset': self.current_dataset,
            'context_history_length': len(self.context_history),
            'recent_queries': [item['query'] for item in self.context_history[-3:]]
        }
    
    def get_suggested_queries(self, dataset_code: Optional[str] = None) -> List[str]:
        """Get suggested queries for the current or specified dataset."""
        suggestions = [
            "What are the top 10 countries by total value?",
            "Show me trends over the last 10 years",
            "Which items have the highest values?",
            "Compare values between regions",
            "What is the average value across all countries?"
        ]
        
        # Add dataset-specific suggestions based on cached metadata
        dataset_context = self.dataset_cache.get(dataset_code or self.current_dataset, {})
        
        if 'unique_item' in dataset_context:
            items = dataset_context['unique_item'][:3]
            for item in items:
                suggestions.append(f"What are the trends for {item}?")
        
        if 'unique_element' in dataset_context:
            elements = dataset_context['unique_element'][:2]
            for element in elements:
                suggestions.append(f"Which countries have the highest {element.lower()}?")
        
        return suggestions[:8]  # Return max 8 suggestions