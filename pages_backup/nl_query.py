"""
Natural language query page for the FAOSTAT Analytics application.

This module provides an interface for asking questions about agricultural data
in natural language and getting AI-powered responses.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional

from components.query_interface import show_query_interface
from features.nl_query_engine import NaturalLanguageQueryEngine

logger = logging.getLogger(__name__)

def show_nl_query_page():
    """Render the natural language query page."""
    
    st.title("💬 Natural Language Queries")
    st.markdown("Ask questions about agricultural data in plain English and get AI-powered insights")
    
    # Check AI availability
    if not check_ai_availability():
        return
    
    # Initialize query engine with proper services
    query_engine = initialize_query_engine()
    
    if not query_engine:
        st.error("❌ Could not initialize natural language query engine")
        return
    
    # Main interface
    render_main_interface(query_engine)
    
    # Query examples and help
    render_query_examples()
    
    # Query analytics
    render_query_analytics()

def check_ai_availability() -> bool:
    """Check if AI services are available."""
    
    openai_service = st.session_state.get('openai_service')
    
    if not openai_service:
        st.error("""
        ❌ **AI Service Not Available**
        
        Natural language queries require an OpenAI API key. Please:
        1. Add your OpenAI API key in the sidebar
        2. Ensure you have sufficient API credits
        3. Check your internet connection
        """)
        
        st.markdown("### 🔧 Quick Setup")
        st.markdown("""
        1. **Get an API key:** Visit [OpenAI's website](https://platform.openai.com/api-keys)
        2. **Add to sidebar:** Use the configuration panel in the sidebar
        3. **Start querying:** Return to this page once configured
        """)
        
        return False
    
    return True

def initialize_query_engine() -> Optional[NaturalLanguageQueryEngine]:
    """Initialize the natural language query engine with required services."""
    
    try:
        # Get required services from session state
        openai_service = st.session_state.get('openai_service')
        faostat_service = st.session_state.get('faostat_service')
        
        if not openai_service:
            st.error("❌ OpenAI service not available. Please configure your API key in the sidebar.")
            return None
        
        if not faostat_service:
            st.error("❌ FAOSTAT service not available. Please restart the application.")
            return None
        
        # Create the query engine with both required services
        query_engine = NaturalLanguageQueryEngine(
            openai_service=openai_service,
            faostat_service=faostat_service
        )
        
        return query_engine
        
    except Exception as e:
        logger.error(f"Error initializing query engine: {str(e)}")
        st.error(f"❌ Error initializing query engine: {str(e)}")
        return None

def render_main_interface(query_engine: NaturalLanguageQueryEngine):
    """Render the main query interface."""
    
    # Get current dataset context - check multiple possible keys safely
    current_dataset = st.session_state.get('current_dataset_code')
    if not current_dataset:
        current_dataset = st.session_state.get('current_dataset')
    
    current_dataset_name = st.session_state.get('current_dataset_name')
    if not current_dataset_name:
        current_dataset_name = st.session_state.get('dataset_name')
    if not current_dataset_name:
        current_dataset_name = current_dataset if current_dataset else 'No dataset selected'
    
    # Safely get dataset DataFrame
    current_dataset_df = None
    for key in ['current_dataset_df', 'dataset_df', 'current_dataset']:
        df_candidate = st.session_state.get(key)
        if df_candidate is not None:
            # Check if it's actually a DataFrame
            if hasattr(df_candidate, 'empty'):  # DataFrame-like object
                current_dataset_df = df_candidate
                break
    
    # Debug info to help identify the issue
    if st.checkbox("🐛 Show Debug Info", help="Show session state keys for debugging"):
        st.write("**Session State Keys:**")
        dataset_keys = [k for k in st.session_state.keys() if 'dataset' in k.lower()]
        st.write(dataset_keys)
        
        st.write("**Current Values:**")
        st.write(f"current_dataset: {current_dataset}")
        st.write(f"current_dataset_name: {current_dataset_name}")
        st.write(f"current_dataset_df type: {type(current_dataset_df)}")
        if current_dataset_df is not None and hasattr(current_dataset_df, 'shape'):
            st.write(f"current_dataset_df shape: {current_dataset_df.shape}")
    
    # Dataset context display
    render_dataset_context(current_dataset, current_dataset_name, current_dataset_df)
    
    # Query interface
    try:
        result = show_query_interface(
            nl_engine=query_engine,
            current_dataset=current_dataset,
            dataset_df=current_dataset_df,
            key_prefix="nl_query"
        )
        
        # Store query result in session state for analytics
        if result and 'query' in result:
            store_query_result(result)
            
    except Exception as e:
        logger.error(f"Error in query interface: {str(e)}")
        st.error(f"❌ Error in query interface: {str(e)}")

def render_dataset_context(dataset_code: Optional[str], dataset_name: str, dataset_df: Optional[pd.DataFrame]):
    """Render information about the current dataset context."""
    
    st.markdown("## 📊 Current Context")
    
    # Check if we have a valid dataset
    has_dataset = False
    if dataset_code is not None and dataset_df is not None:
        try:
            # Safely check if DataFrame is not empty
            has_dataset = isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty
        except Exception:
            has_dataset = False
    
    if has_dataset:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset", dataset_name)
        
        with col2:
            st.metric("Records", f"{len(dataset_df):,}")
        
        with col3:
            if 'Area' in dataset_df.columns:
                st.metric("Countries", dataset_df['Area'].nunique())
            else:
                st.metric("Countries", "N/A")
        
        with col4:
            if 'Year' in dataset_df.columns:
                year_range = f"{dataset_df['Year'].min()}-{dataset_df['Year'].max()}"
                st.metric("Years", year_range)
            else:
                st.metric("Years", "N/A")
        
        # Quick dataset insights
        with st.expander("📈 Quick Dataset Insights"):
            render_quick_insights(dataset_df)
    
    else:
        st.info("""
        📝 **No dataset selected**
        
        For more specific and accurate responses, consider:
        1. Selecting a dataset from the Dataset Browser
        2. Running an analysis first
        3. Using thematic templates
        
        You can still ask general questions about agricultural data and FAOSTAT!
        """)
        
        # Add a quick link to dataset browser
        if st.button("📊 Go to Dataset Browser", type="primary"):
            st.session_state.current_page = "dataset_browser"
            st.rerun()

def render_quick_insights(dataset_df: pd.DataFrame):
    """Render quick insights about the current dataset."""
    
    try:
        # Basic information
        st.markdown("**Dataset Overview:**")
        st.text(f"• {len(dataset_df):,} total records")
        st.text(f"• {len(dataset_df.columns)} columns")
        
        # Geographic coverage
        if 'Area' in dataset_df.columns:
            top_countries = dataset_df['Area'].value_counts().head(3)
            st.markdown("**Top Countries by Records:**")
            for country, count in top_countries.items():
                st.text(f"• {country}: {count:,} records")
        
        # Temporal coverage
        if 'Year' in dataset_df.columns:
            st.markdown("**Temporal Coverage:**")
            st.text(f"• From {dataset_df['Year'].min()} to {dataset_df['Year'].max()}")
            st.text(f"• {dataset_df['Year'].nunique()} unique years")
        
        # Items/commodities
        if 'Item' in dataset_df.columns:
            top_items = dataset_df['Item'].value_counts().head(3)
            st.markdown("**Top Items by Records:**")
            for item, count in top_items.items():
                st.text(f"• {item}: {count:,} records")
        
    except Exception as e:
        logger.error(f"Error generating quick insights: {str(e)}")
        st.text("Could not generate insights for this dataset")

def store_query_result(result: Dict[str, Any]):
    """Store query result for analytics."""
    
    try:
        # Initialize query history if not exists
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        # Add timestamp if not present
        if 'timestamp' not in result:
            from datetime import datetime
            result['timestamp'] = datetime.now()
        
        # Store the result
        st.session_state.query_history.append(result)
        
        # Keep only last 50 queries
        if len(st.session_state.query_history) > 50:
            st.session_state.query_history = st.session_state.query_history[-50:]
            
    except Exception as e:
        logger.error(f"Error storing query result: {str(e)}")

def render_query_examples():
    """Render example queries to help users get started."""
    
    st.markdown("## 💡 Example Questions")
    st.markdown("Need inspiration? Try these example questions:")
    
    # Categorize examples
    example_categories = {
        "🌍 Global Trends": [
            "What are the global trends in wheat production over the last decade?",
            "How has world food security changed since 2010?",
            "Which regions show the fastest agricultural growth?",
        ],
        "🏆 Rankings & Comparisons": [
            "Which countries are the top rice producers?",
            "Compare fertilizer use between developed and developing countries",
            "How does China's agricultural output compare to India's?",
        ],
        "📈 Trend Analysis": [
            "Show me fertilizer use trends in Africa over time",
            "Has corn yield improved in the United States?",
            "What are the patterns in agricultural trade since 2000?",
        ],
        "🎯 Specific Insights": [
            "What factors explain the difference in crop yields between regions?",
            "Which crops show the most volatile production patterns?",
            "How does climate change appear to be affecting agriculture?",
        ]
    }
    
    # Display examples in tabs
    tabs = st.tabs(list(example_categories.keys()))
    
    for tab, (category, examples) in zip(tabs, example_categories.items()):
        with tab:
            for i, example in enumerate(examples):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{example}**")
                
                with col2:
                    if st.button("Try it", key=f"example_{category}_{i}"):
                        st.session_state.nl_query_input = example
                        st.rerun()

def render_query_analytics():
    """Render analytics about query usage and performance."""
    
    query_history = st.session_state.get('query_history', [])
    
    if not query_history:
        return
    
    st.markdown("## 📊 Query Analytics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(query_history))
    
    with col2:
        # Calculate average query length
        avg_length = sum(len(q.get('query', '')) for q in query_history) / len(query_history)
        st.metric("Avg Query Length", f"{avg_length:.0f} chars")
    
    with col3:
        # Count queries today
        from datetime import datetime, timedelta
        today = datetime.now().date()
        queries_today = sum(1 for q in query_history 
                          if q.get('timestamp') and 
                          (q['timestamp'].date() == today if hasattr(q['timestamp'], 'date') else False))
        st.metric("Queries Today", queries_today)
    
    with col4:
        # Most common dataset
        datasets = [q.get('dataset_code', '') for q in query_history if q.get('dataset_code')]
        if datasets:
            from collections import Counter
            most_common = Counter(datasets).most_common(1)[0][0]
            st.metric("Most Queried Dataset", most_common[:20] + "..." if len(most_common) > 20 else most_common)
        else:
            st.metric("Most Queried Dataset", "N/A")
    
    # Query patterns
    with st.expander("📈 Query Patterns"):
        
        # Query length distribution
        query_lengths = [len(q.get('query', '')) for q in query_history]
        
        st.markdown("**Query Length Distribution:**")
        length_bins = {
            "Short (< 50 chars)": sum(1 for l in query_lengths if l < 50),
            "Medium (50-100 chars)": sum(1 for l in query_lengths if 50 <= l < 100),
            "Long (100+ chars)": sum(1 for l in query_lengths if l >= 100)
        }
        
        for category, count in length_bins.items():
            st.text(f"• {category}: {count} queries")
        
        # Common query words
        if query_history:
            all_words = []
            for q in query_history:
                query_text = q.get('query', '')
                if query_text:
                    words = query_text.lower().split()
                    all_words.extend([w for w in words if len(w) > 3])  # Filter short words
            
            if all_words:
                from collections import Counter
                word_counts = Counter(all_words)
                
                st.markdown("**Most Common Query Terms:**")
                for word, count in word_counts.most_common(10):
                    st.text(f"• {word}: {count} times")

def render_query_help():
    """Render help and tips for effective querying."""
    
    with st.expander("❓ Query Help & Tips"):
        
        st.markdown("""
        ### 🎯 How to Ask Effective Questions
        
        **Be Specific:**
        - ✅ "What are wheat production trends in Asia from 2010-2020?"
        - ❌ "Tell me about wheat"
        
        **Include Context:**
        - ✅ "Compare rice yields between China and India"
        - ❌ "Compare rice yields"
        
        **Ask for Specific Analysis:**
        - ✅ "What factors explain fertilizer use differences between regions?"
        - ❌ "Tell me about fertilizers"
        
        ### 📊 Types of Questions You Can Ask
        
        **Trend Questions:**
        - "How has [indicator] changed over time?"
        - "What are the trends in [commodity] production?"
        - "Show me [metric] patterns since [year]"
        
        **Comparison Questions:**
        - "Compare [metric] between [country A] and [country B]"
        - "Which regions have the highest [indicator]?"
        - "How does [country] rank in [metric]?"
        
        **Analysis Questions:**
        - "What factors explain [pattern]?"
        - "What are the implications of [trend]?"
        - "Which variables are correlated with [outcome]?"
        
        ### 🎨 Visualization Requests
        
        - "Show me a chart of [data]"
        - "Create a map visualization of [indicator]"
        - "Plot [variable] over time"
        - "Make a comparison chart between [items]"
        
        ### ⚠️ Current Limitations
        
        - AI responses are based on available dataset information
        - Complex statistical analyses may require manual verification
        - Real-time data updates are not available
        - Responses are limited to agricultural/food-related topics
        """)

if __name__ == "__main__":
    show_nl_query_page()