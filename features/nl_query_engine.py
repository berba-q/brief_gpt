"""
Natural language query engine for FAOSTAT Analytics Application.

This module provides a service class for processing natural language queries
about FAOSTAT data, converting them into structured analysis parameters,
and executing the appropriate data operations.
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
    Uses semantic similarity and fuzzy matching to work across all domains.
    """
    
    def __init__(self):
        # Optional: Initialize sentence transformer for semantic matching
        self.semantic_model = None
        self._init_semantic_model()
        
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
    
    def _init_semantic_model(self):
        """Initialize semantic similarity model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic model loaded successfully")
        except ImportError:
            logger.info("Sentence transformers not available, using fuzzy matching only")
            self.semantic_model = None
    
    def find_best_matches(
        self, 
        query_terms: List[str], 
        available_values: List[str], 
        threshold: float = 70.0,
        max_results: int = 3
    ) -> List[str]:
        """Find best matches using intelligent combination of methods."""
        if not query_terms or not available_values:
            return []
        
        all_matches = []
        
        for query_term in query_terms:
            # Method 1: Fuzzy matching
            fuzzy_matches = self._fuzzy_match(query_term, available_values, threshold)
            
            # Method 2: Pattern matching
            pattern_matches = self._pattern_match(query_term, available_values)
            
            # Method 3: Semantic similarity (if available)
            semantic_matches = []
            if self.semantic_model:
                semantic_matches = self._semantic_match(query_term, available_values, threshold)
            
            # Combine and rank all matches
            combined_matches = self._combine_matches(
                fuzzy_matches, pattern_matches, semantic_matches
            )
            
            all_matches.extend(combined_matches[:max_results])
        
        # Remove duplicates and return top matches
        unique_matches = list(dict.fromkeys(all_matches))
        return unique_matches[:max_results]
    
    def _fuzzy_match(self, query_term: str, available_values: List[str], threshold: float) -> List[Tuple[str, float]]:
        """Fuzzy string matching using rapidfuzz."""
        matches = process.extract(
            query_term,
            available_values,
            scorer=fuzz.WRatio,
            limit=5
        )
        return [(match[0], match[1]) for match in matches if match[1] >= threshold]
    
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
    
    def _semantic_match(self, query_term: str, available_values: List[str], threshold: float) -> List[Tuple[str, float]]:
        """Semantic similarity matching using sentence transformers."""
        if not self.semantic_model:
            return []
        
        try:
            query_embedding = self.semantic_model.encode([query_term])
            value_embeddings = self.semantic_model.encode(available_values)
            similarities = self.semantic_model.similarity(query_embedding, value_embeddings)[0]
            
            matches = []
            for i, similarity in enumerate(similarities):
                score = float(similarity * 100)
                if score >= (threshold * 0.7):
                    matches.append((available_values[i], score))
            
            return sorted(matches, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Semantic matching error: {str(e)}")
            return []
    
    def _combine_matches(self, fuzzy_matches, pattern_matches, semantic_matches) -> List[str]:
        """Combine and rank matches from different methods."""
        scores = {}
        
        # Add fuzzy match scores (weight: 1.0)
        for value, score in fuzzy_matches:
            scores[value] = scores.get(value, 0) + score * 1.0
        
        # Add pattern match scores (weight: 0.8)
        for value, score in pattern_matches:
            scores[value] = scores.get(value, 0) + score * 0.8
        
        # Add semantic match scores (weight: 0.6)
        for value, score in semantic_matches:
            scores[value] = scores.get(value, 0) + score * 0.6
        
        # Sort by combined score
        ranked_values = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [value for value, score in ranked_values]


class NaturalLanguageQueryEngine:
    """
    Engine for processing natural language queries about FAOSTAT data with intelligent matching.
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
        
        # Initialize intelligent matcher
        self.intelligent_matcher = IntelligentFAOSTATMatcher()
        
        # Context and cache
        self.context_history = []
        self.dataset_cache = {}
        self.current_dataset = None
        self.current_dataset_df = None
        
        # Query patterns for different types of questions
        self.query_patterns = self._initialize_query_patterns()
        
        logger.info("Initialized Natural Language Query Engine with intelligent matching")
    
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
                    r"standard\s+deviation\s+(?:of|in)\s+(.+)"
                ],
                "type": "statistical_analysis"
            },
            "factual": {
                "patterns": [
                    r"(?:what|which)\s+(.+?)(?:\s+in\s+(\d{4}|\w+))?",
                    r"(?:how\s+much|how\s+many)\s+(.+?)(?:\s+in\s+(.+))?",
                    r"(?:when\s+did|when\s+was)\s+(.+)",
                    r"(?:where\s+is|where\s+are)\s+(.+)"
                ],
                "type": "factual_query"
            }
        }
    
    def set_current_dataset(self, dataset_code: str, dataset_df: Optional[pd.DataFrame] = None) -> bool:
        """Set the current dataset for query processing."""
        try:
            self.current_dataset = dataset_code
            
            if dataset_df is not None:
                self.current_dataset_df = dataset_df
            else:
                df, metadata = self.faostat_service.get_dataset(dataset_code)
                self.current_dataset_df = df
            
            # Cache dataset metadata for intelligent matching
            if dataset_code not in self.dataset_cache:
                self.dataset_cache[dataset_code] = self._extract_dataset_context(self.current_dataset_df)
            
            logger.info(f"Set current dataset to {dataset_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting current dataset {dataset_code}: {str(e)}")
            return False
    
    def _extract_dataset_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract context information from a dataset for intelligent matching."""
        if df is None or df.empty:
            return {}
        
        context = {
            "columns": list(df.columns),
            "row_count": len(df),
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Extract unique values for key columns (for intelligent matching)
        key_columns = ['Area', 'Item', 'Element', 'Year']
        for col in key_columns:
            if col in df.columns:
                unique_values = df[col].unique().tolist()
                context[f"unique_{col.lower()}"] = unique_values
                
                if col == 'Year':
                    context["year_range"] = [int(df[col].min()), int(df[col].max())]
        
        # Add value statistics if available
        if 'Value' in df.columns:
            context["value_stats"] = {
                "min": float(df['Value'].min()),
                "max": float(df['Value'].max()),
                "mean": float(df['Value'].mean()),
                "median": float(df['Value'].median())
            }
        
        return context
    
    def process_query(
        self,
        query: str,
        dataset_code: Optional[str] = None,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """Process a natural language query with intelligent matching."""
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
            
            # ENHANCED: Use intelligent parsing instead of LLM parsing
            parsed_query = self._parse_query_intelligently(query, query_type)
            
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
        """
        ENHANCED: Parse query using intelligent matching instead of just LLM.
        """
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
            if components.get('regions'):
                available_areas = dataset_context.get('unique_area', [])
                matched_regions = self.intelligent_matcher.find_best_matches(
                    components['regions'], 
                    available_areas,
                    threshold=60.0
                )
                if matched_regions:
                    matched_params['regions'] = matched_regions
            
            # Match items using intelligent matching
            if components.get('items'):
                available_items = dataset_context.get('unique_item', [])
                matched_items = self.intelligent_matcher.find_best_matches(
                    components['items'], 
                    available_items,
                    threshold=50.0
                )
                if matched_items:
                    matched_params['items'] = matched_items
            
            # Match elements using intelligent matching
            if components.get('elements'):
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
                'available_items': available_items[:10] if 'available_items' in locals() else [],
                'available_elements': available_elements if 'available_elements' in locals() else [],
                'matching_method': 'intelligent_fuzzy_semantic'
            }
            
            logger.info(f"Intelligent parsing result: {matched_params}")
            return matched_params
            
        except Exception as e:
            logger.error(f"Error in intelligent parsing: {str(e)}")
            # Fallback to original LLM parsing if intelligent parsing fails
            return self._parse_query_with_llm(query, query_type)
    
    def _extract_query_components(self, query: str) -> Dict[str, List[str]]:
        """Extract basic components from natural language query."""
        query_lower = query.lower()
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(match) for match in re.findall(year_pattern, query)]
        
        # Extract potential regions (capitalized words)
        potential_regions = []
        for word in query.split():
            if word.istitle() and len(word) > 3:
                potential_regions.append(word)
        
        # Extract potential items (common commodities and query words)
        commodity_indicators = ['nitrogen', 'phosphorus', 'potassium', 'wheat', 'rice', 'maize', 'corn', 'cattle', 'chicken', 'milk']
        potential_items = []
        for indicator in commodity_indicators:
            if indicator in query_lower:
                potential_items.append(indicator)
        
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
        """ENHANCED: Clean parameters using intelligent matching."""
        dataset_context = self.dataset_cache.get(self.current_dataset, {})
        cleaned = {}
        
        # Extract and validate years
        if "years" in params and params["years"]:
            years = params["years"]
            if isinstance(years, list):
                valid_years = [y for y in years if isinstance(y, int) and 1960 <= y <= 2030]
                if valid_years:
                    cleaned["year_range"] = (min(valid_years), max(valid_years))
        
        # ENHANCED: Use intelligent matching for items
        if "items" in params and params["items"]:
            items = params["items"]
            if isinstance(items, list):
                available_items = dataset_context.get('unique_item', [])
                matched_items = self.intelligent_matcher.find_best_matches(
                    items, available_items, threshold=50.0
                )
                if matched_items:
                    cleaned["items"] = matched_items
                    logger.info(f"Intelligent item matching: {items} → {matched_items}")
        
        # ENHANCED: Use intelligent matching for elements
        if "elements" in params and params["elements"]:
            elements = params["elements"]
            if isinstance(elements, list):
                available_elements = dataset_context.get('unique_element', [])
                matched_elements = self.intelligent_matcher.find_best_matches(
                    elements, available_elements, threshold=50.0
                )
                if matched_elements:
                    cleaned["elements"] = matched_elements
                    logger.info(f"Intelligent element matching: {elements} → {matched_elements}")
        
        # ENHANCED: Use intelligent matching for regions
        if "regions" in params and params["regions"]:
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
        """Execute the parsed query on the dataset."""
        try:
            if "error" in parsed_query:
                return {"error": parsed_query["error"]}
            
            df = self.current_dataset_df.copy()
            
            # Apply filters based on parsed parameters
            filtered_df = self._apply_filters(df, parsed_query)
            
            # Determine query type and execute appropriate analysis
            query_type = parsed_query.get("query_type", "general_query")
            
            if query_type == "ranking_comparison":
                return self._execute_ranking_query(filtered_df, parsed_query)
            elif query_type == "time_series":
                return self._execute_trend_query(filtered_df, parsed_query)
            elif query_type == "statistical_analysis":
                return self._execute_statistical_query(filtered_df, parsed_query)
            elif query_type == "factual_query":
                return self._execute_factual_query(filtered_df, parsed_query)
            else:
                return self._execute_general_query(filtered_df, parsed_query)
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"error": f"Failed to execute query: {str(e)}"}
    
    def _apply_filters(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataset based on parsed query parameters."""
        filtered_df = df.copy()
        
        # Apply year filter
        if "year_range" in parsed_query and "Year" in filtered_df.columns:
            start_year, end_year = parsed_query["year_range"]
            filtered_df = filtered_df[
                (filtered_df["Year"] >= start_year) & (filtered_df["Year"] <= end_year)
            ]
        
        # Apply region filter
        if "regions" in parsed_query and "Area" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Area"].isin(parsed_query["regions"])]
        
        # Apply item filter
        if "items" in parsed_query and "Item" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Item"].isin(parsed_query["items"])]
        
        # Apply element filter
        if "elements" in parsed_query and "Element" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Element"].isin(parsed_query["elements"])]
        
        return filtered_df
    
    def _execute_ranking_query(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ranking/comparison queries."""
        if df.empty or "Value" not in df.columns:
            return {"error": "No data available for comparison"}
        
        # Group by Area and sum values
        if "Area" in df.columns:
            ranking = df.groupby("Area")["Value"].sum().sort_values(ascending=False)
            
            result = {
                "type": "ranking",
                "top_10": ranking.head(10).to_dict(),
                "bottom_10": ranking.tail(10).to_dict(),
                "total_entities": len(ranking),
                "data_summary": {
                    "highest": {"name": ranking.index[0], "value": float(ranking.iloc[0])},
                    "lowest": {"name": ranking.index[-1], "value": float(ranking.iloc[-1])},
                    "average": float(ranking.mean()),
                    "median": float(ranking.median())
                }
            }
            
            return result
        
        return {"error": "Cannot perform ranking without geographic dimension"}
    
    def _execute_trend_query(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trend/time series queries."""
        if df.empty or "Year" not in df.columns or "Value" not in df.columns:
            return {"error": "No time series data available"}
        
        # Aggregate by year
        trend_data = df.groupby("Year")["Value"].sum().sort_index()
        
        # Calculate growth rates
        growth_rates = trend_data.pct_change() * 100
        
        # Calculate overall statistics
        first_year = trend_data.index[0]
        last_year = trend_data.index[-1]
        first_value = trend_data.iloc[0]
        last_value = trend_data.iloc[-1]
        
        total_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
        years_span = last_year - first_year
        cagr = ((last_value / first_value) ** (1 / years_span) - 1) * 100 if years_span > 0 and first_value > 0 else 0
        
        result = {
            "type": "trend",
            "time_series": trend_data.to_dict(),
            "growth_rates": growth_rates.dropna().to_dict(),
            "trend_summary": {
                "first_year": int(first_year),
                "last_year": int(last_year),
                "first_value": float(first_value),
                "last_value": float(last_value),
                "total_change_percent": float(total_change),
                "cagr_percent": float(cagr),
                "average_annual_growth": float(growth_rates.mean()) if not growth_rates.empty else 0
            }
        }
        
        return result
    
    def _execute_statistical_query(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis queries."""
        if df.empty or "Value" not in df.columns:
            return {"error": "No numerical data available for statistical analysis"}
        
        values = df["Value"].dropna()
        
        result = {
            "type": "statistical",
            "statistics": {
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "sum": float(values.sum()),
                "percentiles": {
                    "25th": float(values.quantile(0.25)),
                    "75th": float(values.quantile(0.75))
                }
            }
        }
        
        # Add correlation analysis if multiple numeric columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()
            result["correlations"] = correlations.to_dict()
        
        return result
    
    def _execute_factual_query(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute factual/informational queries."""
        if df.empty:
            return {"error": "No data available"}
        
        # Get basic facts about the filtered data
        facts = {
            "type": "factual",
            "data_overview": {
                "total_records": len(df),
                "date_range": None,
                "countries_covered": None,
                "items_covered": None,
                "elements_covered": None
            }
        }
        
        # Add specific facts based on available columns
        if "Year" in df.columns:
            facts["data_overview"]["date_range"] = {
                "start": int(df["Year"].min()),
                "end": int(df["Year"].max())
            }
        
        if "Area" in df.columns:
            facts["data_overview"]["countries_covered"] = int(df["Area"].nunique())
            facts["data_overview"]["top_countries"] = df["Area"].value_counts().head(5).to_dict()
        
        if "Item" in df.columns:
            facts["data_overview"]["items_covered"] = int(df["Item"].nunique())
            facts["data_overview"]["top_items"] = df["Item"].value_counts().head(5).to_dict()
        
        if "Element" in df.columns:
            facts["data_overview"]["elements_covered"] = int(df["Element"].nunique())
            facts["data_overview"]["available_elements"] = df["Element"].unique().tolist()
        
        if "Value" in df.columns:
            facts["data_overview"]["value_summary"] = {
                "total": float(df["Value"].sum()),
                "average": float(df["Value"].mean()),
                "records_with_data": int(df["Value"].count())
            }
        
        return facts
    
    def _execute_general_query(self, df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general queries that don't fit other categories."""
        # Provide a general summary of the filtered data
        summary = {
            "type": "general",
            "filtered_data_summary": {
                "total_records": len(df),
                "columns": list(df.columns)
            }
        }
        
        # Add summary statistics for each column type
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                summary["filtered_data_summary"][f"{col}_stats"] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
            else:
                summary["filtered_data_summary"][f"{col}_unique_count"] = int(df[col].nunique())
        
        return summary
    
    def _generate_response(self, query: str, result: Dict[str, Any], parsed_query: Dict[str, Any]) -> str:
        """Generate a human-readable response based on query results."""
        try:
            if "error" in result:
                return f"I couldn't process your query: {result['error']}"
            
            # Use OpenAI to generate a fact-based response
            fact_sheet = {
                "query": query,
                "dataset": self.current_dataset,
                "filtered_data": result,
                "query_parameters": parsed_query
            }
            
            # Create a prompt for fact-based response
            prompt = get_fact_based_response_prompt(query, fact_sheet)
            
            messages = [
                {"role": "system", "content": "You are an expert agricultural data analyst who provides clear, factual responses based only on the provided data."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.openai_service._call_openai_api(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=400,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I found some information about your query, but had trouble formatting the response. The analysis shows: {str(result)[:200]}..."
    
    def _add_to_context(self, query: str) -> None:
        """Add query to context history."""
        self.context_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "dataset": self.current_dataset
        })
        
        # Keep only recent context
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def get_suggested_queries(self, dataset_code: Optional[str] = None) -> List[str]:
        """Get suggested queries for a dataset."""
        if dataset_code:
            self.set_current_dataset(dataset_code)
        
        if not self.current_dataset:
            return [
                "What datasets are available?",
                "Show me food production data for 2020",
                "Which countries produce the most rice?"
            ]
        
        # Generate suggestions based on dataset content
        context = self.dataset_cache.get(self.current_dataset, {})
        suggestions = []
        
        # Basic data exploration
        suggestions.append(f"What data is available in the {self.current_dataset} dataset?")
        
        # Time-based suggestions
        if "year_range" in context:
            start_year, end_year = context["year_range"]
            suggestions.extend([
                f"Show trends from {start_year} to {end_year}",
                f"What changed between {start_year} and {end_year}?"
            ])
        
        # Geographic suggestions
        if "unique_area" in context and context["unique_area"]:
            top_areas = context["unique_area"][:3]
            suggestions.extend([
                f"Compare {', '.join(top_areas)}",
                "Which countries have the highest values?",
                "Show regional comparison"
            ])
        
        # Item-based suggestions
        if "unique_item" in context and context["unique_item"]:
            top_items = context["unique_item"][:3]
            suggestions.extend([
                f"What are the trends for {top_items[0]}?",
                f"Compare production of {', '.join(top_items)}"
            ])
        
        return suggestions[:8] # Limit to 8 suggestions
    
    def clear_context(self) -> None:
        """Clear conversation context."""
        self.context_history = []
        logger.info("Cleared conversation context")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context."""
        return {
            "current_dataset": self.current_dataset,
            "context_history_length": len(self.context_history),
            "recent_queries": [item["query"] for item in self.context_history[-3:]],
            "available_datasets": list(self.dataset_cache.keys())
        }