"""
OpenAI service for integrating with GPT models - UPDATED VERSION.

This module provides a service class for interacting with OpenAI's API,
generating insights, and handling various prompt-based tasks while
implementing cost-saving strategies and dataset theme detection.
"""

from openai import OpenAI
import json
import time
import hashlib
import logging
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

from utils.prompt_templates import (
    get_insights_prompt,
    get_interactive_analysis_prompt,
    get_visualization_suggestion_prompt,
    get_structured_query_prompt
)

# Set up logger
logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Service for interacting with OpenAI's GPT models.
    
    This class provides methods to:
    - Generate analytical insights from data
    - Answer interactive queries about data
    - Suggest appropriate visualizations
    - Parse natural language queries
    - Implements caching, throttling, and other cost-saving strategies
    
    Attributes:
        client (OpenAI): OpenAI client instance
        model (str): GPT model to use
        max_tokens (int): Maximum tokens for generation
        temperature (float): Temperature setting (0.0-1.0)
        cache_enabled (bool): Whether to use response caching
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1500,
        temperature: float = 0.7,
        cache_enabled: bool = True
    ):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4)
            max_tokens: Maximum tokens for generation
            temperature: Temperature setting (0.0-1.0)
            cache_enabled: Whether to use response caching
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        
        # Set up OpenAI client (modern API)
        self.client = OpenAI(api_key=api_key)
        
        # Rate limiting settings
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds between requests
        
        # Response cache
        self._cache = {}
        
        # Token usage tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        logger.info(f"Initialized OpenAI service with model: {model}")
    
    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a cache key from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Cache key
        """
        # Convert messages to a stable string representation
        message_str = json.dumps(messages, sort_keys=True)
        
        # Generate an MD5 hash
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Get a response from the cache if it exists.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Optional[str]: Cached response or None
        """
        if not self.cache_enabled:
            return None
        
        return self._cache.get(cache_key)
    
    def _save_to_cache(self, cache_key: str, response: str) -> None:
        """
        Save a response to the cache.
        
        Args:
            cache_key: Cache key
            response: Response to cache
        """
        if not self.cache_enabled:
            return
        
        self._cache[cache_key] = response
    
    def _throttle_requests(self) -> None:
        """
        Throttle requests to stay within rate limits.
        This adds a small delay between requests if needed.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            # Sleep for the remaining time
            time.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _update_token_usage(self, usage_data) -> None:
        """
        Update token usage statistics.
        
        Args:
            usage_data: Token usage data from API response
        """
        # Handle both old and new API response formats
        if hasattr(usage_data, 'prompt_tokens'):
            # New API format
            self.token_usage["prompt_tokens"] += usage_data.prompt_tokens
            self.token_usage["completion_tokens"] += usage_data.completion_tokens
            self.token_usage["total_tokens"] += usage_data.total_tokens
        elif isinstance(usage_data, dict):
            # Old API format (fallback)
            self.token_usage["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
            self.token_usage["completion_tokens"] += usage_data.get("completion_tokens", 0)
            self.token_usage["total_tokens"] += usage_data.get("total_tokens", 0)
    
    def _call_openai_api(
        self, 
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True
    ) -> str:
        """
        Call the OpenAI API with error handling and caching.
        
        Args:
            messages: List of message dictionaries
            model: Optional override for model
            temperature: Optional override for temperature
            max_tokens: Optional override for max_tokens
            use_cache: Whether to check and use cache for this request
            
        Returns:
            str: Generated text response
        """
        # Use provided parameters or fall back to instance defaults
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Generate cache key and check cache if enabled
        cache_key = self._get_cache_key(messages) if use_cache else None
        
        if use_cache and cache_key:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.debug("Returning cached response")
                return cached_response
        
        # Throttle requests to stay within rate limits
        self._throttle_requests()
        
        try:
            # Call the OpenAI API using the modern client
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Update token usage statistics
            if hasattr(response, 'usage'):
                self._update_token_usage(response.usage)
            
            # Save to cache if enabled
            if use_cache and cache_key:
                self._save_to_cache(cache_key, response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    def generate_insights(
        self, 
        dataset_name: str,
        summary_stats: Dict[str, Any],
        figures_descriptions: List[str],
        model_type: str = "comprehensive",
        dataset_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate insights for an analytical brief using GPT.
        
        Args:
            dataset_name: Name of the dataset
            summary_stats: Summary statistics of the dataset
            figures_descriptions: List of figure descriptions
            model_type: Type of model to use (comprehensive or standard)
            dataset_metadata: Original FAOSTAT metadata for theme detection
            
        Returns:
            dict: Generated insights and narrative sections
        """
        try:
            # Select model configuration based on model type
            if model_type == "standard":
                model = "gpt-4"
                max_tokens = 800
                temperature = 0.5
            else:  # comprehensive
                model = self.model
                max_tokens = self.max_tokens
                temperature = self.temperature
            
            # Create prompt for the LLM with metadata support
            prompt = get_insights_prompt(
                dataset_name=dataset_name, 
                summary_stats=summary_stats, 
                figures_descriptions=figures_descriptions,
                dataset_metadata=dataset_metadata
            )
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": "You are an expert agricultural data analyst specialized in FAO data interpretation. Always respond with valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the OpenAI API
            content = self._call_openai_api(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Parse the response - expect JSON format
            try:
                # Clean the content to extract JSON
                json_str = content.strip()
                
                # Handle common JSON formatting issues
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    # Try to extract from any code block
                    json_str = content.split("```")[1].split("```")[0].strip()
                
                # Try to find JSON-like content if no code blocks
                if not json_str.startswith('{'):
                    # Look for the first { and last }
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        json_str = content[start:end+1]
                
                # Parse the JSON
                insights = json.loads(json_str)
                
                # Validate that we have the expected structure
                expected_keys = ['highlights', 'background', 'global_trends', 'regional_analysis', 'explanatory_notes']
                if not all(key in insights for key in expected_keys):
                    logger.warning("Incomplete insights structure received from OpenAI")
                    # Fill in missing keys
                    for key in expected_keys:
                        if key not in insights:
                            insights[key] = f"Information not available for {key.replace('_', ' ')}"
                
                return insights
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Error parsing JSON from OpenAI response: {str(json_error)}")
                logger.error(f"Raw content (first 500 chars): {content[:500]}...")
                
                # If JSON parsing fails, try to extract meaningful content
                insights = {
                    "highlights": [
                        "Analysis completed but formatting issues occurred",
                        "Raw AI response contains valuable insights",
                        "Manual review of analysis recommended"
                    ],
                    "background": self._extract_section(content, "background") or content[:500] + "...",
                    "global_trends": self._extract_section(content, "trends") or "See background section for analysis.",
                    "regional_analysis": self._extract_section(content, "regional") or "See background section for analysis.", 
                    "explanatory_notes": f"Note: JSON parsing failed. Response length: {len(content)} characters. Please review raw output for insights."
                }
                
                return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                "highlights": [f"Error generating insights: {str(e)}"],
                "background": "An error occurred while generating insights with the LLM.",
                "global_trends": "Analysis could not be completed due to an error.",
                "regional_analysis": "Analysis could not be completed due to an error.",
                "explanatory_notes": f"Error details: {str(e)}"
            }
    
    def _extract_section(self, content: str, section_keyword: str) -> Optional[str]:
        """
        Try to extract a section from unstructured content.
        
        Args:
            content: Raw content from AI
            section_keyword: Keyword to look for
            
        Returns:
            Optional[str]: Extracted section or None
        """
        try:
            content_lower = content.lower()
            keyword_lower = section_keyword.lower()
            
            # Look for the section keyword
            start_idx = content_lower.find(keyword_lower)
            if start_idx == -1:
                return None
            
            # Find the start of the actual content (after the keyword)
            start_idx = content.find(':', start_idx)
            if start_idx == -1:
                start_idx = content_lower.find(keyword_lower) + len(keyword_lower)
            else:
                start_idx += 1
            
            # Find the end (next section or end of content)
            section_keywords = ['background', 'trends', 'regional', 'analysis', 'notes']
            end_idx = len(content)
            
            for keyword in section_keywords:
                if keyword != section_keyword:
                    keyword_idx = content_lower.find(keyword, start_idx)
                    if keyword_idx != -1:
                        end_idx = min(end_idx, keyword_idx)
            
            # Extract and clean the section
            section = content[start_idx:end_idx].strip()
            
            # Clean up common formatting issues
            section = section.replace('\n\n', ' ').replace('\n', ' ')
            section = ' '.join(section.split())  # Normalize whitespace
            
            return section[:300] + "..." if len(section) > 300 else section
            
        except Exception:
            return None
    
    def generate_interactive_analysis(
        self, 
        dataset_df: Optional[pd.DataFrame],
        question: str,
        context: Optional[Dict[str, Any]] = None,
        model_type: str = "standard"
    ) -> str:
        """
        Generate an interactive analysis based on a user question about the dataset.
        
        Args:
            dataset_df: The dataset being analyzed (or None if not needed)
            question: User's question about the data
            context: Additional context information
            model_type: Type of model to use (comprehensive or standard)
            
        Returns:
            str: Analysis response
        """
        try:
            # Select model configuration based on model type
            if model_type == "comprehensive":
                model = self.model
                max_tokens = self.max_tokens
                temperature = self.temperature
            else:  # standard
                model = "gpt-4"
                max_tokens = 800
                temperature = 0.5
            
            # Create dataset overview if dataset is provided
            dataset_info = {}
            if dataset_df is not None:
                dataset_info = {
                    "columns": list(dataset_df.columns),
                    "row_count": len(dataset_df),
                    "sample_data": dataset_df.head(5).to_dict(orient='records')
                }
                
                # Add information about specific columns
                if 'Area' in dataset_df.columns:
                    dataset_info["available_areas"] = list(dataset_df['Area'].unique())[:20]
                
                if 'Item' in dataset_df.columns:
                    dataset_info["available_items"] = list(dataset_df['Item'].unique())[:20]
                
                if 'Year' in dataset_df.columns:
                    dataset_info["year_range"] = [int(dataset_df['Year'].min()), int(dataset_df['Year'].max())]
            
            # Create prompt for the question
            prompt = get_interactive_analysis_prompt(question, dataset_info, context)
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": "You are an expert agricultural data analyst who explains data clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the OpenAI API
            response = self._call_openai_api(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating interactive analysis: {str(e)}")
            return f"Error analyzing the data: {str(e)}"
    
    def suggest_visualizations(
        self, 
        dataset_df: pd.DataFrame,
        model_type: str = "standard"
    ) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations for the dataset.
        
        Args:
            dataset_df: The dataset to analyze
            model_type: Type of model to use (comprehensive or standard)
            
        Returns:
            list: List of suggested visualization approaches
        """
        try:
            # Select model configuration
            if model_type == "comprehensive":
                model = self.model
                max_tokens = self.max_tokens
                temperature = 0.7
            else:  # standard
                model = "gpt-4"
                max_tokens = 800
                temperature = 0.5
            
            # Create a dataset overview
            columns_info = {col: str(dataset_df[col].dtype) for col in dataset_df.columns}
            column_nunique = {col: int(dataset_df[col].nunique()) for col in dataset_df.columns}
            
            # Create prompt for visualization suggestions
            prompt = get_visualization_suggestion_prompt(columns_info, column_nunique, len(dataset_df))
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": "You are an expert data visualization specialist who provides practical and insightful visualization recommendations in JSON format."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the OpenAI API
            content = self._call_openai_api(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Parse the response - expect JSON format
            try:
                # Find JSON content in the response
                json_str = content
                
                # Check for code blocks
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                
                suggestions = json.loads(json_str)
                return suggestions
            except Exception as json_error:
                logger.error(f"Error parsing JSON from OpenAI response: {str(json_error)}")
                # Return the raw content with an error note
                return [{"type": "Error", "columns": [], "insight": "Failed to parse JSON response", "note": content}]
            
        except Exception as e:
            logger.error(f"Error generating visualization suggestions: {str(e)}")
            return [{"type": "Error", "columns": [], "insight": f"Error generating visualization suggestions: {str(e)}"}]
    
    def parse_natural_language_query(
        self, 
        query: str,
        dataset_context: Dict[str, Any],
        model_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Parse a natural language query into structured parameters.
        
        Args:
            query: Natural language query
            dataset_context: Context about the current dataset
            model_type: Type of model to use (comprehensive or standard)
            
        Returns:
            dict: Structured query parameters
        """
        try:
            # Select model configuration
            if model_type == "comprehensive":
                model = self.model
                max_tokens = 800
                temperature = 0.3
            else:  # standard
                model = "gpt-4"
                max_tokens = 500
                temperature = 0.3
            
            # Create prompt for query parsing
            prompt = get_structured_query_prompt(query, dataset_context)
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": "You are an expert data analyst who can parse natural language queries into structured JSON parameters."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the OpenAI API
            content = self._call_openai_api(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Parse the response - expect JSON format
            try:
                # Find JSON content in the response
                json_str = content
                
                # Check for code blocks
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                
                parsed_params = json.loads(json_str)
                return parsed_params
            except Exception as json_error:
                logger.error(f"Error parsing JSON from OpenAI response: {str(json_error)}")
                return {
                    "error": f"Failed to parse query: {str(json_error)}",
                    "raw_response": content
                }
            
        except Exception as e:
            logger.error(f"Error parsing natural language query: {str(e)}")
            return {
                "error": f"Error parsing query: {str(e)}"
            }
    
    def get_token_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics about token usage.
        
        Returns:
            dict: Token usage statistics
        """
        return self.token_usage.copy()
    
    def estimate_cost(self, model: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate the cost of API usage.
        
        Args:
            model: Model to estimate cost for (defaults to instance model)
            
        Returns:
            dict: Cost estimates in USD
        """
        model = model or self.model
        
        # Updated cost per 1K tokens (as of 2024 pricing)
        cost_per_1k = {
            "gpt-4": {
                "prompt": 0.03,
                "completion": 0.06
            },
            "gpt-4-turbo": {
                "prompt": 0.01,
                "completion": 0.03
            },
            "gpt-3.5-turbo": {
                "prompt": 0.0015,
                "completion": 0.002
            }
        }
        
        # Get the cost rates or use gpt-3.5-turbo as fallback
        model_cost = cost_per_1k.get(model, cost_per_1k["gpt-3.5-turbo"])
        
        # Calculate costs
        prompt_cost = (self.token_usage["prompt_tokens"] / 1000) * model_cost["prompt"]
        completion_cost = (self.token_usage["completion_tokens"] / 1000) * model_cost["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "model": model
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache = {}
        logger.info("Cleared OpenAI response cache")