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

from models.data_models import NaturalLanguageQuery, AnalysisFilter
from services.openai_service import OpenAIService
from services.faostat_service import FAOStatService
from utils.prompt_templates import get_structured_query_prompt, get_fact_based_response_prompt

# Set up logger
logger = logging.getLogger(__name__)

class NaturalLanguageQueryEngine:
    """
    Engine for processing natural language queries about FAOSTAT data.
    
    This class provides methods to:
    - Parse natural language queries into structured parameters
    - Execute data queries based on parsed parameters
    - Generate human-readable responses to questions
    - Handle context and follow-up questions
    
    Attributes:
        openai_service (OpenAIService): Service for LLM processing
        faostat_service (FAOStatService): Service for data access
        context_history (list): History of previous queries for context
        dataset_cache (dict): Cache of dataset metadata for faster processing
    """
    
    def __init__(
        self,
        openai_service: OpenAIService,
        faostat_service: FAOStatService,
        max_context_history: int = 5
    ):
        """
        Initialize the natural language query engine.
        
        Args:
            openai_service: OpenAI service for LLM processing
            faostat_service: FAOSTAT service for data access
            max_context_history: Maximum number of queries to keep in context
        """
        self.openai_service = openai_service
        self.faostat_service = faostat_service
        self.max_context_history = max_context_history
        
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
        """
        Set the current dataset for query processing.
        
        Args:
            dataset_code: Code of the dataset to set as current
            dataset_df: Optional DataFrame of the dataset
            
        Returns:
            True if dataset was set successfully, False otherwise
        """
        try:
            self.current_dataset = dataset_code
            
            if dataset_df is not None:
                self.current_dataset_df = dataset_df
            else:
                # Load dataset if not provided
                df, metadata = self.faostat_service.get_dataset(dataset_code)
                self.current_dataset_df = df
            
            # Cache dataset metadata
            if dataset_code not in self.dataset_cache:
                self.dataset_cache[dataset_code] = self._extract_dataset_context(self.current_dataset_df)
            
            logger.info(f"Set current dataset to {dataset_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting current dataset {dataset_code}: {str(e)}")
            return False
    
    def _extract_dataset_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract context information from a dataset."""
        if df is None or df.empty:
            return {}
        
        context = {
            "columns": list(df.columns),
            "row_count": len(df),
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Extract unique values for key columns
        key_columns = ['Area', 'Item', 'Element', 'Year']
        for col in key_columns:
            if col in df.columns:
                unique_values = df[col].unique().tolist()
                context[f"unique_{col.lower()}"] = unique_values[:50]  # Limit to 50 for performance
                
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
        """
        Process a natural language query and return structured results.
        
        Args:
            query: Natural language query
            dataset_code: Optional dataset code (uses current if not provided)
            use_context: Whether to use conversation context
            
        Returns:
            Dictionary with query results and metadata
        """
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
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query based on patterns."""
        query_lower = query.lower()
        
        for category, config in self.query_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    return config["type"]
        
        return "general_query"
    
    def _parse_query_with_llm(self, query: str, query_type: str) -> Dict[str, Any]:
        """Parse query using LLM to extract structured parameters."""
        try:
            # Get dataset context
            dataset_context = self.dataset_cache.get(self.current_dataset, {})
            
            # Use OpenAI service to parse the query
            parsed_params = self.openai_service.parse_natural_language_query(
                query, dataset_context, model_type="standard"
            )
            
            # Add query type and clean up parameters
            if not parsed_params.get("error"):
                parsed_params["query_type"] = query_type
                parsed_params = self._clean_parsed_parameters(parsed_params)
            
            return parsed_params
            
        except Exception as e:
            logger.error(f"Error parsing query with LLM: {str(e)}")
            return {"error": f"Failed to parse query: {str(e)}"}
    
    def _clean_parsed_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate parsed parameters."""
        cleaned = {}
        
        # Extract and validate years
        if "years" in params and params["years"]:
            years = params["years"]
            if isinstance(years, list):
                valid_years = [y for y in years if isinstance(y, int) and 1960 <= y <= 2030]
                if valid_years:
                    cleaned["year_range"] = (min(valid_years), max(valid_years))
        
        # Extract and validate regions
        if "regions" in params and params["regions"]:
            regions = params["regions"]
            if isinstance(regions, list):
                available_regions = self.dataset_cache.get(self.current_dataset, {}).get("unique_area", [])
                valid_regions = [r for r in regions if r in available_regions]
                if valid_regions:
                    cleaned["regions"] = valid_regions
        
        # Extract and validate items
        if "items" in params and params["items"]:
            items = params["items"]
            if isinstance(items, list):
                available_items = self.dataset_cache.get(self.current_dataset, {}).get("unique_item", [])
                valid_items = [i for i in items if i in available_items]
                if valid_items:
                    cleaned["items"] = valid_items
        
        # Extract and validate elements
        if "elements" in params and params["elements"]:
            elements = params["elements"]
            if isinstance(elements, list):
                available_elements = self.dataset_cache.get(self.current_dataset, {}).get("unique_element", [])
                valid_elements = [e for e in elements if e in available_elements]
                if valid_elements:
                    cleaned["elements"] = valid_elements
        
        # Extract metrics and visualization type
        if "metrics" in params:
            cleaned["metrics"] = params["metrics"]
        
        if "visualization_type" in params:
            cleaned["visualization_type"] = params["visualization_type"]
        
        return cleaned
    
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
        
        # Group by Area (or other relevant dimension) and sum values
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
        
        return suggestions[:8]  # Return top 8 suggestions
    
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