"""
Sidebar component for FAOSTAT Analytics Application.

This module provides the main navigation sidebar with page selection,
system status, and configuration options.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional

from config.constants import THEMATIC_TEMPLATES

logger = logging.getLogger(__name__)

def create_sidebar() -> str:
    """
    Create the main application sidebar with navigation.
    
    Returns:
        str: Selected page identifier
    """
    
    with st.sidebar:
        # App logo and title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #2E7D32; font-size: 1.5rem; margin: 0;'>
                🌾 FAOSTAT Analytics
            </h1>
            <p style='color: #666; font-size: 0.9rem; margin: 0;'>
                Agricultural Data Intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        st.subheader("📋 Navigation")
        
        page = st.selectbox(
            "Select a page:",
            [
                ("home", "🏠 Home"),
                ("dataset_browser", "📊 Dataset Browser"),
                ("analysis", "🔬 Analysis"),
                ("thematic_templates", "📋 Thematic Templates"),
                ("nl_query", "🤖 Natural Language Queries"),
                ("about", "ℹ️ About")
            ],
            format_func=lambda x: x[1],
            key="page_selector"
        )
        
        selected_page = page[0]
        
        st.markdown("---")
        
        # Quick stats
        if 'faostat_service' in st.session_state:
            _show_quick_stats()
        
        # Current dataset info
        if 'current_dataset' in st.session_state and st.session_state.current_dataset:
            _show_current_dataset_info()
        
        # System status
        _show_system_status()
        
        # Settings
        with st.expander("⚙️ Settings"):
            _show_settings()
    
    return selected_page

def _show_quick_stats():
    """Show quick statistics about the application."""
    
    st.subheader("📈 Quick Stats")
    
    try:
        # Show template count
        if 'template_manager' in st.session_state:
            templates = st.session_state.template_manager.get_all_templates()
            st.metric("Templates", len(templates))
        
        # Show bookmarks count  
        if 'template_manager' in st.session_state:
            bookmarks = st.session_state.template_manager.get_all_bookmarks()
            st.metric("Bookmarks", len(bookmarks))
        
        # Show current dataset
        if 'current_dataset' in st.session_state and st.session_state.current_dataset:
            st.info(f"📊 Current: {st.session_state.current_dataset}")
    
    except Exception as e:
        st.error(f"Error loading stats: {str(e)}")

def _show_current_dataset_info():
    """Show information about the currently selected dataset."""
    
    st.subheader("📊 Current Dataset")
    
    current_dataset = st.session_state.current_dataset
    st.write(f"**{current_dataset}**")
    
    # Show dataset info if available
    if 'current_dataset_df' in st.session_state and st.session_state.current_dataset_df is not None:
        df = st.session_state.current_dataset_df
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        
        # Show year range if available
        if 'Year' in df.columns:
            year_min = int(df['Year'].min())
            year_max = int(df['Year'].max())
            st.write(f"📅 {year_min} - {year_max}")
    
    # Clear dataset button
    if st.button("🗑️ Clear Dataset", use_container_width=True):
        st.session_state.current_dataset = None
        st.session