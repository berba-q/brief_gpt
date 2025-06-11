"""
Natural language query page for the FAOSTAT Analytics application.

This module provides an interface for asking questions about agricultural data
in natural language and getting AI-powered responses.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional

from components.query_interface import render_query_interface, render_advanced_query_options
from features.nl_query_engine import NLQueryEngine

logger = logging.getLogger(__name__)

def render():
    """Render the natural language query page."""
    
    st.title("💬 Natural Language Queries")
    st.markdown("Ask questions about agricultural data in plain English and get AI-powered insights")
    
    # Check AI availability
    if not check_ai_availability():
        return
    
    # Initialize query engine
    query_engine = NLQueryEngine()
    
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

def render_main_interface(query_engine):
    """Render the main query interface."""
    
    # Get current dataset context
    current_dataset = st.session_state.get('current_dataset')
    current_dataset_name = st.session_state.get('current_dataset_name', 'No dataset selected')
    
    # Dataset context display
    render_dataset_context(current_dataset, current_dataset_name)
    
    # Query interface
    render_query_interface(current_dataset, current_dataset_name)
    
    # Advanced options
    render_advanced_query_options()

def render_dataset_context(dataset_df: Optional[pd.DataFrame], dataset_name: str):
    """Render information about the current dataset context."""
    
    st.markdown("## 📊 Current Context")
    
    if dataset_df is not None and not dataset_df.empty:
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
        avg_length = sum(len(q['query']) for q in query_history) / len(query_history)
        st.metric("Avg Query Length", f"{avg_length:.0f} chars")
    
    with col3:
        # Count queries today
        from datetime import datetime, timedelta
        today = datetime.now().date()
        queries_today = sum(1 for q in query_history if q['timestamp'].date() == today)
        st.metric("Queries Today", queries_today)
    
    with col4:
        # Most common dataset
        datasets = [q['dataset'] for q in query_history if q['dataset']]
        if datasets:
            most_common = max(set(datasets), key=datasets.count)
            st.metric("Most Queried Dataset", most_common[:20] + "..." if len(most_common) > 20 else most_common)
        else:
            st.metric("Most Queried Dataset", "N/A")
    
    # Query patterns
    with st.expander("📈 Query Patterns"):
        
        # Query length distribution
        query_lengths = [len(q['query']) for q in query_history]
        
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
                words = q['query'].lower().split()
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

def render_query_feedback():
    """Render interface for query feedback and improvement."""
    
    st.markdown("## 💬 Feedback & Suggestions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Improve This Feature")
        
        feedback_type = st.selectbox(
            "Feedback Type",
            options=[
                "Feature Request",
                "Query Accuracy Issue", 
                "Interface Improvement",
                "General Feedback"
            ]
        )
        
        feedback_text = st.text_area(
            "Your Feedback",
            placeholder="Tell us how we can improve the natural language query feature...",
            height=100
        )
        
        if st.button("📨 Submit Feedback"):
            # In a real implementation, this would send feedback to developers
            st.success("Thank you for your feedback! We appreciate your input.")
    
    with col2:
        st.markdown("### 🔧 Troubleshooting")
        
        st.markdown("""
        **Common Issues:**
        
        **Slow Responses:**
        - Check internet connection
        - Try shorter, more specific queries
        - Verify OpenAI API key is working
        
        **Inaccurate Responses:**
        - Ensure dataset is loaded correctly
        - Try rephrasing your question
        - Be more specific about what you want
        
        **No Response:**
        - Check OpenAI API key configuration
        - Verify you have API credits remaining
        - Try refreshing the page
        
        **Need Help:**
        - Use the example questions as templates
        - Start with simple queries and build complexity
        - Check the Query Help section above
        """)

def render_advanced_features():
    """Render advanced natural language query features."""
    
    st.markdown("## 🚀 Advanced Features")
    
    # Multi-step queries
    with st.expander("🔄 Multi-Step Analysis"):
        st.markdown("""
        **Chain Multiple Questions:**
        
        You can ask follow-up questions that build on previous responses:
        
        1. "What are the top wheat producing countries?"
        2. "How have these countries' yields changed over time?"
        3. "What factors might explain the differences?"
        
        The AI will remember context from previous questions in your session.
        """)
        
        if st.button("🧪 Try Multi-Step Example"):
            st.session_state.nl_query_input = "What are the top 5 wheat producing countries?"
            st.rerun()
    
    # Query templates
    with st.expander("📋 Query Templates"):
        st.markdown("**Quick Query Templates:**")
        
        templates = {
            "Trend Analysis": "How has [INDICATOR] changed in [REGION] from [START_YEAR] to [END_YEAR]?",
            "Country Ranking": "Which countries have the highest [METRIC] in [YEAR]?",
            "Regional Comparison": "Compare [INDICATOR] between [REGION_A] and [REGION_B]",
            "Factor Analysis": "What factors explain the differences in [OUTCOME] between countries?",
            "Future Outlook": "Based on historical trends, what might happen to [INDICATOR] in the future?"
        }
        
        for template_name, template_text in templates.items():
            st.markdown(f"**{template_name}:**")
            st.code(template_text)
    
    # Data export from queries
    with st.expander("💾 Export Query Results"):
        st.markdown("""
        **Export Options:**
        
        - **Text Export:** Download query and response as text file
        - **Data Export:** Export any data referenced in the response
        - **Visualization Export:** Save any charts mentioned in responses
        - **Report Generation:** Include query results in formal reports
        
        Use the export buttons that appear after each query response.
        """)

# Main render function with all sections
def render():
    """Render the complete natural language query page."""
    
    st.title("💬 Natural Language Queries")
    st.markdown("Ask questions about agricultural data in plain English and get AI-powered insights")
    
    # Check AI availability
    if not check_ai_availability():
        return
    
    # Initialize query engine
    try:
        query_engine = NLQueryEngine()
    except Exception as e:
        st.error(f"Error initializing query engine: {str(e)}")
        return
    
    # Main interface
    render_main_interface(query_engine)
    
    st.markdown("---")
    
    # Query examples and help
    render_query_examples()
    
    st.markdown("---")
    
    # Advanced features
    render_advanced_features()
    
    st.markdown("---")
    
    # Query help
    render_query_help()
    
    st.markdown("---")
    
    # Analytics (only if there's history)
    if st.session_state.get('query_history'):
        render_query_analytics()
        st.markdown("---")
    
    # Feedback
    render_query_feedback()