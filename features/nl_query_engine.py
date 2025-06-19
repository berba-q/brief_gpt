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
    
    FIXED: Handles PyTorch tensor device issues properly.
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
        """Initialize semantic similarity model if available - FIXED for device issues."""
        try:
            # Import torch first to set device properly
            import torch
            
            # Force CPU usage to avoid device mismatch issues
            if torch.cuda.is_available():
                logger.info("CUDA available but forcing CPU for compatibility")
            
            # Set device to CPU explicitly
            device = torch.device('cpu')
            
            # Import and initialize sentence transformers with CPU device
            from sentence_transformers import SentenceTransformer
            
            # Initialize model with explicit device specification
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # Double-check all model components are on CPU
            if hasattr(self.semantic_model, '_modules'):
                for module in self.semantic_model._modules.values():
                    if hasattr(module, 'to'):
                        module.to(device)
            
            logger.info("Semantic model loaded successfully on CPU device")
            
        except ImportError as e:
            logger.info(f"Sentence transformers not available: {e}")
            self.semantic_model = None
        except RuntimeError as e:
            logger.error(f"PyTorch runtime error during model initialization: {e}")
            logger.info("Falling back to fuzzy matching only")
            self.semantic_model = None
        except Exception as e:
            logger.error(f"Unexpected error initializing semantic model: {e}")
            logger.info("Falling back to fuzzy matching only")
            self.semantic_model = None
    
    def find_best_matches(
        self, 
        query_terms: List[str], 
        available_values: List[str], 
        threshold: float = 70.0,
        max_results: int = 3
    ) -> List[str]:
        """Find best matches using intelligent combination of methods."""
        
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
            
            # Get semantic matches (only if model is available)
            semantic_matches = []
            if self.semantic_model:
                try:
                    semantic_matches = self._semantic_match(query_term, available_values, threshold)
                except Exception as e:
                    logger.warning(f"Semantic matching failed for '{query_term}': {e}")
                    # Don't fail completely, just skip semantic matching
                    semantic_matches = []
            
            # Combine matches for this term
            term_matches = self._combine_matches(fuzzy_matches, pattern_matches, semantic_matches)
            
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
    
    def _semantic_match(self, query_term: str, available_values: List[str], threshold: float) -> List[Tuple[str, float]]:
        """Semantic similarity matching using sentence transformers - FIXED for device issues."""
        if not self.semantic_model:
            return []
        
        try:
            # Ensure we're working on CPU
            import torch
            device = torch.device('cpu')
            
            # Move model to CPU if it's not already there
            if hasattr(self.semantic_model, 'to'):
                self.semantic_model.to(device)
            
            # Encode with explicit device handling
            with torch.no_grad():  # Disable gradient computation for inference
                query_embedding = self.semantic_model.encode([query_term], 
                                                           convert_to_tensor=True,
                                                           device='cpu')
                value_embeddings = self.semantic_model.encode(available_values[:100],  # Limit to prevent memory issues
                                                            convert_to_tensor=True,
                                                            device='cpu')
                
                # Ensure tensors are on the same device
                query_embedding = query_embedding.to(device)
                value_embeddings = value_embeddings.to(device)
                
                # Compute similarities using torch.nn.functional.cosine_similarity
                similarities = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    value_embeddings.unsqueeze(1),
                    dim=2
                ).squeeze()
                
                # Convert to numpy for processing
                similarities = similarities.cpu().numpy()
            
            matches = []
            for i, similarity in enumerate(similarities):
                if i >= len(available_values):  # Safety check
                    break
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
    
    FIXED: Better error handling and device management.
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
        """Execute the parsed query on the dataset."""
        try:
            # This is a placeholder - implement your query execution logic here
            return {
                "success": True,
                "data": "Query execution placeholder",
                "parsed_params": parsed_query
            }
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {"error": f"Query execution failed: {str(e)}"}
    
    def _generate_response(self, query: str, result: Dict[str, Any], parsed_query: Dict[str, Any]) -> str:
        """Generate a natural language response from query results."""
        try:
            # This is a placeholder - implement your response generation logic here
            return f"Based on your query '{query}', here are the results..."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating the response."
    
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