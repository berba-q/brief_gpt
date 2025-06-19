"""
Query interface component for FAOSTAT Analytics Application.

This module provides a reusable UI component for natural language query processing,
including query input, suggestion display, result visualization, and query history.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from features.nl_query_engine import NaturalLanguageQueryEngine
from utils.visualization import VisualizationGenerator

# Set up logger
logger = logging.getLogger(__name__)

def show_query_interface(
    nl_engine: NaturalLanguageQueryEngine,
    current_dataset: Optional[str] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    key_prefix: str = "query"
) -> Dict[str, Any]:
    """
    Display a natural language query interface.
    
    Args:
        nl_engine: Natural language query engine instance
        current_dataset: Currently selected dataset code
        dataset_df: Current dataset DataFrame
        key_prefix: Prefix for Streamlit widget keys
        
    Returns:
        Dictionary with query results and metadata
    """
    
    # Initialize session state for query history
    if f"{key_prefix}_history" not in st.session_state:
        st.session_state[f"{key_prefix}_history"] = []
    
    if f"{key_prefix}_suggestions_visible" not in st.session_state:
        st.session_state[f"{key_prefix}_suggestions_visible"] = True
    
    # Set current dataset in engine if provided
    if current_dataset and dataset_df is not None:
        nl_engine.set_current_dataset(current_dataset)
    
    # Main query interface
    st.subheader("🤖 Ask Questions About Your Data")
    
    # Dataset status
    if current_dataset:
        st.info(f"📊 Currently analyzing: **{current_dataset}** dataset")
    else:
        st.warning("⚠️ No dataset selected. Please select a dataset first.")
        return {}
    
    # Query input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about the data:",
            placeholder="e.g., Which countries have the highest rice production in 2020?",
            key=f"{key_prefix}_input",
            help="Type your question in natural language. I can help with trends, comparisons, rankings, and statistics."
        )
    
    with col2:
        ask_button = st.button(
            "🔍 Ask",
            key=f"{key_prefix}_ask_button",
            type="primary",
            use_container_width=True
        )
    
    # Query suggestions section
    if st.session_state[f"{key_prefix}_suggestions_visible"]:
        with st.expander("💡 Suggested Questions", expanded=False):
            try:
                suggestions = nl_engine.get_suggested_queries(current_dataset)
                if suggestions:
                    st.write("Click on any suggestion to use it as your query:")
                    
                    # Display suggestions in a grid
                    cols = st.columns(2)
                    for i, suggestion in enumerate(suggestions):
                        col_idx = i % 2
                        with cols[col_idx]:
                            if st.button(
                                suggestion,
                                key=f"{key_prefix}_suggestion_{i}",
                                use_container_width=True
                            ):
                                st.session_state[f"{key_prefix}_input"] = suggestion
                                st.rerun()
                else:
                    st.info("No suggestions available for this dataset.")
            except Exception as e:
                st.error(f"Error loading suggestions: {str(e)}")
    
    # Process query
    result = {}
    if ask_button and query.strip():
        with st.spinner("🔍 Processing your question..."):
            try:
                result = nl_engine.process_query(query, current_dataset)
                
                # Add to history
                history_entry = {
                    "query": query,
                    "timestamp": datetime.now(),
                    "dataset": current_dataset,
                    "result": result
                }
                st.session_state[f"{key_prefix}_history"].append(history_entry)
                
                # Keep only last 10 queries
                if len(st.session_state[f"{key_prefix}_history"]) > 10:
                    st.session_state[f"{key_prefix}_history"] = st.session_state[f"{key_prefix}_history"][-10:]
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Query processing error: {str(e)}")
                return {}
    
    # Display results
    if result:
        _display_query_results(result, key_prefix)
    
    # Query history section
    if st.session_state[f"{key_prefix}_history"]:
        with st.expander(f"📝 Query History ({len(st.session_state[f'{key_prefix}_history'])} queries)", expanded=False):
            _display_query_history(st.session_state[f"{key_prefix}_history"], key_prefix)
    
    # Settings section
    with st.expander("⚙️ Query Settings", expanded=False):
        _display_query_settings(nl_engine, key_prefix)
    
    return result

def _display_query_results(result: Dict[str, Any], key_prefix: str) -> None:
    """Display the results of a natural language query."""
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Display the response
    if "response" in result and result["response"]:
        st.success("✅ **Query Result:**")
        st.write(result["response"])
    
    # Display query metadata
    with st.expander("🔍 Query Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Information:**")
            st.write(f"- Query Type: {result.get('query_type', 'Unknown')}")
            st.write(f"- Dataset: {result.get('dataset_code', 'Unknown')}")
            st.write(f"- Processed: {result.get('timestamp', 'Unknown')}")
        
        with col2:
            if "parsed_query" in result:
                st.write("**Parsed Parameters:**")
                parsed = result["parsed_query"]
                
                if "year_range" in parsed:
                    st.write(f"- Years: {parsed['year_range'][0]} - {parsed['year_range'][1]}")
                
                if "regions" in parsed and parsed["regions"]:
                    st.write(f"- Regions: {', '.join(parsed['regions'][:3])}...")
                
                if "items" in parsed and parsed["items"]:
                    st.write(f"- Items: {', '.join(parsed['items'][:3])}...")
                
                if "elements" in parsed and parsed["elements"]:
                    st.write(f"- Elements: {', '.join(parsed['elements'][:3])}...")
    
    # Display detailed results based on query type
    if "result" in result and isinstance(result["result"], dict):
        query_result = result["result"]
        result_type = query_result.get("type", "unknown")
        
        if result_type == "ranking":
            _display_ranking_results(query_result, key_prefix)
        elif result_type == "trend":
            _display_trend_results(query_result, key_prefix)
        elif result_type == "statistical":
            _display_statistical_results(query_result, key_prefix)
        elif result_type == "factual":
            _display_factual_results(query_result, key_prefix)

def _display_ranking_results(ranking_data: Dict[str, Any], key_prefix: str) -> None:
    """Display ranking/comparison results."""
    
    st.subheader("📊 Ranking Results")
    
    # Summary statistics
    if "data_summary" in ranking_data:
        summary = ranking_data["data_summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Top Performer", summary.get("highest", {}).get("name", "N/A"))
        with col2:
            st.metric("Highest Value", f"{summary.get('highest', {}).get('value', 0):,.0f}")
        with col3:
            st.metric("Average", f"{summary.get('average', 0):,.0f}")
        with col4:
            st.metric("Total Countries", summary.get("total_entities", 0))
    
    # Top 10 and Bottom 10
    col1, col2 = st.columns(2)
    
    with col1:
        if "top_10" in ranking_data:
            st.write("**🏆 Top 10:**")
            top_df = pd.DataFrame(list(ranking_data["top_10"].items()), columns=["Country", "Value"])
            top_df["Value"] = top_df["Value"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    with col2:
        if "bottom_10" in ranking_data:
            st.write("**📉 Bottom 10:**")
            bottom_df = pd.DataFrame(list(ranking_data["bottom_10"].items()), columns=["Country", "Value"])
            bottom_df["Value"] = bottom_df["Value"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(bottom_df, use_container_width=True, hide_index=True)

def _display_trend_results(trend_data: Dict[str, Any], key_prefix: str) -> None:
    """Display trend/time series results."""
    
    st.subheader("📈 Trend Analysis")
    
    # Summary statistics
    if "trend_summary" in trend_data:
        summary = trend_data["trend_summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Period", f"{summary.get('first_year', 'N/A')} - {summary.get('last_year', 'N/A')}")
        with col2:
            change = summary.get("total_change_percent", 0)
            st.metric("Total Change", f"{change:+.1f}%")
        with col3:
            cagr = summary.get("cagr_percent", 0)
            st.metric("CAGR", f"{cagr:+.1f}%")
        with col4:
            avg_growth = summary.get("average_annual_growth", 0)
            st.metric("Avg Annual Growth", f"{avg_growth:+.1f}%")
    
    # Time series chart
    if "time_series" in trend_data:
        ts_data = trend_data["time_series"]
        if ts_data:
            df = pd.DataFrame(list(ts_data.items()), columns=["Year", "Value"])
            df["Year"] = pd.to_numeric(df["Year"])
            df = df.sort_values("Year")
            
            st.line_chart(df.set_index("Year"), use_container_width=True)

def _display_statistical_results(stats_data: Dict[str, Any], key_prefix: str) -> None:
    """Display statistical analysis results."""
    
    st.subheader("📊 Statistical Analysis")
    
    if "statistics" in stats_data:
        stats = stats_data["statistics"]
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Count", f"{stats.get('count', 0):,}")
        with col2:
            st.metric("Mean", f"{stats.get('mean', 0):,.2f}")
        with col3:
            st.metric("Median", f"{stats.get('median', 0):,.2f}")
        with col4:
            st.metric("Std Dev", f"{stats.get('std', 0):,.2f}")
        
        # Range and percentiles
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Minimum", f"{stats.get('min', 0):,.2f}")
        with col2:
            st.metric("25th Percentile", f"{stats.get('percentiles', {}).get('25th', 0):,.2f}")
        with col3:
            st.metric("75th Percentile", f"{stats.get('percentiles', {}).get('75th', 0):,.2f}")
        with col4:
            st.metric("Maximum", f"{stats.get('max', 0):,.2f}")
    
    # Correlations if available
    if "correlations" in stats_data:
        with st.expander("🔗 Correlation Analysis", expanded=False):
            corr_data = stats_data["correlations"]
            if isinstance(corr_data, dict):
                st.write("Correlation matrix:")
                corr_df = pd.DataFrame(corr_data)
                st.dataframe(corr_df.style.format("{:.3f}"), use_container_width=True)

def _display_factual_results(factual_data: Dict[str, Any], key_prefix: str) -> None:
    """Display factual query results."""
    
    st.subheader("📋 Data Overview")
    
    if "data_overview" in factual_data:
        overview = factual_data["data_overview"]
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{overview.get('total_records', 0):,}")
        with col2:
            if overview.get('countries_covered'):
                st.metric("Countries", overview['countries_covered'])
        with col3:
            if overview.get('items_covered'):
                st.metric("Items", overview['items_covered'])
        
        # Date range
        if "date_range" in overview:
            date_range = overview["date_range"]
            st.info(f"📅 Data covers: {date_range.get('start', 'N/A')} - {date_range.get('end', 'N/A')}")
        
        # Top items/countries
        col1, col2 = st.columns(2)
        
        with col1:
            if "top_countries" in overview:
                st.write("**🌍 Most Frequent Countries:**")
                countries_df = pd.DataFrame(
                    list(overview["top_countries"].items()),
                    columns=["Country", "Records"]
                )
                st.dataframe(countries_df, use_container_width=True, hide_index=True)
        
        with col2:
            if "top_items" in overview:
                st.write("**📦 Most Frequent Items:**")
                items_df = pd.DataFrame(
                    list(overview["top_items"].items()),
                    columns=["Item", "Records"]
                )
                st.dataframe(items_df, use_container_width=True, hide_index=True)

def _display_query_history(history: List[Dict[str, Any]], key_prefix: str) -> None:
    """Display query history."""
    
    for i, entry in enumerate(reversed(history)):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**Q:** {entry['query']}")
                if entry.get('result', {}).get('response'):
                    response = entry['result']['response']
                    if len(response) > 100:
                        response = response[:100] + "..."
                    st.write(f"**A:** {response}")
            
            with col2:
                st.write(f"*{entry['dataset']}*")
            
            with col3:
                timestamp = entry['timestamp']
                if isinstance(timestamp, datetime):
                    st.write(f"*{timestamp.strftime('%H:%M')}*")
                
                if st.button("🔄", key=f"{key_prefix}_rerun_{i}", help="Re-run this query"):
                    st.session_state[f"{key_prefix}_input"] = entry['query']
                    st.rerun()

def _display_query_settings(nl_engine: NaturalLanguageQueryEngine, key_prefix: str) -> None:
    """Display query engine settings."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear History", key=f"{key_prefix}_clear_history"):
            st.session_state[f"{key_prefix}_history"] = []
            st.success("Query history cleared!")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Context", key=f"{key_prefix}_clear_context"):
            nl_engine.clear_context()
            st.success("Conversation context cleared!")
    
    # Context information
    try:
        context_summary = nl_engine.get_context_summary()
        
        st.write("**Engine Status:**")
        st.write(f"- Current Dataset: {context_summary.get('current_dataset', 'None')}")
        st.write(f"- Context History: {context_summary.get('context_history_length', 0)} items")
        
        if context_summary.get('recent_queries'):
            st.write("- Recent Queries:")
            for query in context_summary['recent_queries']:
                st.write(f"  • {query[:50]}...")
        
    except Exception as e:
        st.warning(f"Could not load context summary: {str(e)}")

def create_simple_query_input(
    placeholder: str = "Ask a question about the data...",
    key: str = "simple_query"
) -> str:
    """
    Create a simple query input widget.
    
    Args:
        placeholder: Placeholder text for the input
        key: Unique key for the widget
        
    Returns:
        The input query string
    """
    return st.text_input(
        "Query:",
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed"
    )

def show_query_suggestions(
    suggestions: List[str],
    key_prefix: str = "suggestions"
) -> Optional[str]:
    """
    Show query suggestions as clickable buttons.
    
    Args:
        suggestions: List of suggested queries
        key_prefix: Prefix for button keys
        
    Returns:
        Selected suggestion or None
    """
    selected_suggestion = None
    
    if suggestions:
        st.write("**💡 Try these questions:**")
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:6]):  # Show max 6 suggestions
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(
                    suggestion,
                    key=f"{key_prefix}_{i}",
                    use_container_width=True,
                    help="Click to use this query"
                ):
                    selected_suggestion = suggestion
    
    return selected_suggestion