"""
Prompt templates for OpenAI service.

This module provides functions for generating consistent prompt templates
for various use cases in the FAOSTAT Analytics Application with enhanced
dataset theme detection.
"""

import json
from typing import Dict, List, Any, Optional

def get_insights_prompt(
    dataset_name: str,
    summary_stats: Dict[str, Any],
    figures_descriptions: List[str],
    dataset_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a prompt for dataset insights with theme detection.
    
    Args:
        dataset_name: Name of the dataset
        summary_stats: Summary statistics of the dataset
        figures_descriptions: List of figure descriptions
        dataset_metadata: Original FAOSTAT metadata for theme detection
        
    Returns:
        str: Formatted prompt
    """
    
    # Detect dataset theme from metadata
    dataset_theme = detect_dataset_theme(dataset_metadata, dataset_name)
    
    # Create the expected JSON structure as a string (not f-string)
    expected_json_structure = """{
  "highlights": ["point 1", "point 2", "point 3", "point 4"],
  "background": "text...",
  "global_trends": "text...",
  "regional_analysis": "text...",
  "explanatory_notes": "text..."
}"""
    
    prompt = f"""You are an expert agricultural data analyst at the FAO (Food and Agriculture Organization). 
Your task is to write an analytical brief based on the following FAOSTAT dataset: "{dataset_name}".

DATASET THEME: {dataset_theme}

Here's a summary of the dataset and analysis:
{json.dumps(summary_stats, indent=2)}

The analysis includes the following figures:
{json.dumps(figures_descriptions, indent=2)}

Please generate the following sections for an analytical brief focused on {dataset_theme}:

1. HIGHLIGHTS: 3-4 bullet points highlighting the most important findings related to {dataset_theme} (keep each highlight under 50 words)
2. BACKGROUND: A brief introduction to the dataset and its context in {dataset_theme} (100-150 words)
3. GLOBAL TRENDS: Analysis of the global trends shown in the data related to {dataset_theme} (150-200 words)
4. REGIONAL ANALYSIS: Comparison between regions or countries in the context of {dataset_theme} (150-200 words)
5. EXPLANATORY NOTES: Any technical notes or data limitations to be aware of (100 words)

Format your response as JSON with the following structure:
{expected_json_structure}

Ensure your analysis is specifically tailored to {dataset_theme} and uses appropriate terminology for this domain.
"""
    return prompt

def detect_dataset_theme(metadata: Optional[Dict[str, Any]], dataset_name: str) -> str:
    """
    Detect the theme/domain of a dataset from FAOSTAT metadata.
    
    Args:
        metadata: Original FAOSTAT dataset metadata
        dataset_name: Name of the dataset
        
    Returns:
        str: Detected theme
    """
    
    if not metadata:
        return _fallback_theme_detection(dataset_name)
    
    # Extract relevant fields for theme detection
    dataset_code = metadata.get('DatasetCode', '')
    description = metadata.get('DatasetDescription', '').lower()
    topic = metadata.get('Topic', '').lower()
    dataset_name_lower = dataset_name.lower()
    
    # Theme detection rules based on FAOSTAT patterns
    theme_keywords = {
        'production and crops': [
            'crop', 'production', 'yield', 'harvest', 'agricultural production',
            'livestock', 'animal production', 'aquaculture', 'forestry'
        ],
        'trade and markets': [
            'trade', 'import', 'export', 'market', 'price', 'value', 'quantity traded'
        ],
        'food security and nutrition': [
            'food security', 'nutrition', 'food balance', 'dietary', 'undernourishment',
            'calorie', 'protein', 'food supply', 'availability'
        ],
        'environment and sustainability': [
            'emission', 'greenhouse gas', 'climate', 'environmental', 'sustainability',
            'carbon', 'land use', 'deforestation', 'biodiversity'
        ],
        'inputs and resources': [
            'fertilizer', 'pesticide', 'machinery', 'irrigation', 'land', 'water',
            'input', 'resource', 'technology'
        ],
        'population and demographics': [
            'population', 'demographic', 'rural', 'urban', 'employment', 'labor'
        ],
        'economic indicators': [
            'economic', 'gdp', 'income', 'poverty', 'investment', 'expenditure'
        ]
    }
    
    # Check dataset code patterns (FAOSTAT uses specific codes)
    code_themes = {
        'production and crops': ['QCL', 'QA', 'QP', 'QL', 'QV'],
        'trade and markets': ['TCL', 'TM', 'TP', 'PP'],
        'food security and nutrition': ['FBS', 'FS', 'FB'],
        'environment and sustainability': ['GT', 'GE', 'EF', 'EE', 'EL', 'EN'],
        'inputs and resources': ['RFN', 'RP', 'RI', 'RL', 'RT'],
        'population and demographics': ['OA', 'IC'],
        'economic indicators': ['IC', 'IG']
    }
    
    # First check dataset code
    for theme, codes in code_themes.items():
        if any(dataset_code.startswith(code) for code in codes):
            return theme
    
    # Then check keywords in description and topic
    combined_text = f"{description} {topic} {dataset_name_lower}"
    
    theme_scores = {}
    for theme, keywords in theme_keywords.items():
        score = sum(1 for keyword in keywords if keyword in combined_text)
        if score > 0:
            theme_scores[theme] = score
    
    # Return theme with highest score
    if theme_scores:
        return max(theme_scores, key=theme_scores.get)
    
    # Fallback to dataset name analysis
    return _fallback_theme_detection(dataset_name)

def _fallback_theme_detection(dataset_name: str) -> str:
    """Fallback theme detection based on dataset name."""
    
    name_lower = dataset_name.lower()
    
    if any(word in name_lower for word in ['crop', 'livestock', 'production', 'yield']):
        return 'production and crops'
    elif any(word in name_lower for word in ['trade', 'import', 'export']):
        return 'trade and markets'
    elif any(word in name_lower for word in ['food', 'nutrition', 'security']):
        return 'food security and nutrition'
    elif any(word in name_lower for word in ['emission', 'environment', 'climate']):
        return 'environment and sustainability'
    elif any(word in name_lower for word in ['fertilizer', 'input', 'resource']):
        return 'inputs and resources'
    else:
        return 'agricultural data analysis'

def get_interactive_analysis_prompt(
    question: str,
    dataset_info: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a prompt for interactive analysis.
    
    Args:
        question: User's question
        dataset_info: Information about the dataset
        context: Additional context
        
    Returns:
        str: Formatted prompt
    """
    context_str = json.dumps(context, indent=2) if context else "No additional context provided."
    
    prompt = f"""You are an expert data analyst examining a FAOSTAT agricultural dataset.

DATASET INFORMATION:
{json.dumps(dataset_info, indent=2)}

ADDITIONAL CONTEXT:
{context_str}

USER QUESTION: {question}

Please analyze the dataset information and answer the user's question. If the question requires analysis beyond what can be determined from the provided information, explain what additional data or processing would be needed to fully answer the question.

Keep your response under 300 words and focus on addressing the specific question directly. Base your answer ONLY on the data provided - do not make up or infer information not present in the dataset information.
"""
    return prompt

def get_visualization_suggestion_prompt(
    columns_info: Dict[str, str],
    column_nunique: Dict[str, int],
    row_count: int
) -> str:
    """
    Generate a prompt for visualization suggestions.
    
    Args:
        columns_info: Column names and their types
        column_nunique: Number of unique values per column
        row_count: Total number of rows
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""You are an expert data visualization specialist.
            
DATASET INFORMATION:
- Columns and their types: {json.dumps(columns_info)}
- Number of unique values per column: {json.dumps(column_nunique)}
- Total rows: {row_count}

Based on this FAOSTAT agricultural dataset, suggest 3-5 effective visualizations that would reveal meaningful patterns and insights. 

For each suggestion:
1. Name the visualization type
2. Explain which columns to use
3. Describe what pattern or insight it might reveal
4. Suggest any specific settings (e.g., grouping, filtering, aggregation)

Format your response as a JSON array where each element is an object with keys: "type", "columns", "insight", "settings"
"""
    return prompt

def get_structured_query_prompt(
    query: str,
    dataset_context: Dict[str, Any]
) -> str:
    """
    Generate a prompt for parsing natural language queries.
    
    Args:
        query: Natural language query
        dataset_context: Context about the current dataset
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""Parse this natural language query about FAOSTAT agricultural data into structured parameters.

Current dataset information:
{json.dumps(dataset_context, indent=2)}

User query: "{query}"

Extract the following information from the query:
1. Time period (years)
2. Regions or countries
3. Items (crops, livestock, etc.)
4. Elements (production, yield, etc.)
5. Specific metrics or calculations requested
6. Type of visualization that would be appropriate

Return ONLY a JSON object with these extracted parameters. If the query doesn't specify a parameter, set it to null.
VERY IMPORTANT: Only include regions, items, elements, and years that are present in the dataset information provided.
"""
    return prompt

def get_rankings_analysis_prompt(
    top_countries: List[Dict[str, Any]],
    bottom_countries: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> str:
    """
    Generate a prompt for country rankings analysis.
    
    Args:
        top_countries: Data for top countries
        bottom_countries: Data for bottom countries
        stats: Statistics comparing top and bottom
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""You are an expert agricultural data analyst at the FAO (Food and Agriculture Organization).
Analyze these country rankings and provide insights:

TOP 10 COUNTRIES:
{json.dumps(top_countries, indent=2)}

BOTTOM 10 COUNTRIES:
{json.dumps(bottom_countries, indent=2)}

STATISTICS:
- Top 10 average: {stats.get('top10_mean', 0):.2f}
- Bottom 10 average: {stats.get('bottom10_mean', 0):.2f}
- Ratio between top and bottom: {stats.get('top10_to_bottom10_ratio', 0):.1f}x
- Global average: {stats.get('global_mean', 0):.2f}
- Global median: {stats.get('global_median', 0):.2f}
- Total countries analyzed: {stats.get('total_countries', 0)}

Please provide insights about these country rankings:
1. What patterns do you see in the top performing countries? (geographical, economic, etc.)
2. What challenges might the bottom-performing countries be facing?
3. What key factors might explain the large gap between top and bottom countries?
4. What policy recommendations might help close this gap?

Keep your analysis focused on agricultural and economic factors. Format your response in Markdown with clear sections.
"""
    return prompt

def get_thematic_analysis_prompt(
    template_name: str,
    template_prompt: str,
    dataset_name: str,
    summary_stats: Dict[str, Any],
    figures_descriptions: List[str]
) -> str:
    """
    Generate a prompt for thematic analysis.
    
    Args:
        template_name: Name of the template
        template_prompt: Template-specific prompt
        dataset_name: Name of the dataset
        summary_stats: Summary statistics
        figures_descriptions: List of figure descriptions
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""You are analyzing FAOSTAT data on {template_name}.

DATASET: {dataset_name}

DATASET INFORMATION:
{json.dumps(summary_stats, indent=2)}

VISUALIZATIONS:
{json.dumps(figures_descriptions, indent=2)}

Based on the visualizations and data, please provide:

{template_prompt}

Format your response as a focused thematic analysis with clear sections and insights.
"""
    return prompt

def get_fact_based_response_prompt(
    query: str,
    fact_sheet: Dict[str, Any]
) -> str:
    """
    Generate a prompt for fact-based responses.
    
    Args:
        query: User's query
        fact_sheet: Verified facts to base response on
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""Answer this FAOSTAT agricultural data query: "{query}"

IMPORTANT: Base your answer ONLY on the following verified data facts. 
DO NOT make up or infer any information not explicitly present in these facts.

Data Facts:
{json.dumps(fact_sheet, indent=2)}

Your response MUST:
1. Be completely factual and based ONLY on the provided data
2. Express uncertainty if the facts don't provide a clear answer
3. Use specific numbers from the facts, with proper units
4. Stay focused on answering the specific query
5. Note any limitations in the available data relevant to the query

Also, mention which visualizations would be most helpful for understanding this data.
"""
    return prompt