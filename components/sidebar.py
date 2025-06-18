"""
Sidebar component for FAOSTAT Analytics Application.

This module provides the main navigation sidebar with page selection,
system status, and configuration options with fast loading.
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
        
        # Use radio buttons for clean navigation
        page_options = [
            ("home", "🏠 Home"),
            ("dataset_browser", "📊 Dataset Browser"),
            ("analysis", "🔬 Analysis"),
            ("thematic_templates", "📋 Thematic Templates"),
            ("nl_query", "🤖 Natural Language Queries"),
            ("about", "ℹ️ About")
        ]
        
        # Render modern button-based menu
        st.markdown("""
        <style>
            div.stButton > button {
                border-radius: 8px;
                padding: 0.75em 1em;
                margin-bottom: 0.5em;
                border: none;
                font-size: 1.0em;
                text-align: left;
                width: 100%;
                transition: all 0.2s ease-in-out;
            }
            div.stButton > button:hover {
                background-color: #b2dfdb !important;
                color: #004d40 !important;
                transform: scale(1.02);
            }
        </style>
        """, unsafe_allow_html=True)

        for page_id, page_label in page_options:
            is_selected = st.session_state.get('current_page', 'home') == page_id
            button_style = (
                "background-color: #e0f2f1; color: #004d40; font-weight: bold;"
                if is_selected else
                "background-color: transparent; color: #333;"
            )

            if st.button(page_label, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown("---")
        
        # Current dataset info (if any)
        _show_current_dataset_info()
        
        # Quick stats (fast loading)
        _show_quick_stats_fast()
        
        # System status
        _show_system_status()
        
        # Settings
        with st.expander("⚙️ Settings"):
            _show_settings()
    
    return st.session_state.current_page

def _show_settings():
    """Show application settings and configuration options."""
    
    # OpenAI API Key
    current_key = st.session_state.get('openai_api_key', '')
    openai_key = st.text_input(
        "OpenAI API Key",
        value=current_key,
        type="password",
        help="Enter your OpenAI API key to enable AI features.",
        key="openai_key_input"
    )
    
    if openai_key != current_key:
        st.session_state.openai_api_key = openai_key
        # Reinitialize OpenAI service when key changes
        if openai_key:
            try:
                from services.openai_service import OpenAIService
                from features.nl_query_engine import NaturalLanguageQueryEngine
                
                config = st.session_state.get('config')
                if config:
                    model_config = config.get_model_config('comprehensive')
                    openai_service = OpenAIService(
                        api_key=openai_key,
                        model=model_config.get('model', 'gpt-4'),
                        max_tokens=model_config.get('max_tokens', 1500),
                        temperature=model_config.get('temperature', 0.7)
                    )
                    st.session_state.openai_service = openai_service
                    
                    # Also initialize NL engine
                    if 'faostat_service' in st.session_state:
                        nl_engine = NaturalLanguageQueryEngine(
                            openai_service, 
                            st.session_state.faostat_service
                        )
                        st.session_state.nl_engine = nl_engine
                    
                    st.success("✅ OpenAI service initialized!")
            except Exception as e:
                st.error(f"Error initializing OpenAI: {str(e)}")
    
    # Cache management
    st.markdown("### 🗄️ Cache Management")
    if st.button("🧹 Clear Cache", use_container_width=True):
        # Clear FAOSTAT service cache
        if 'faostat_service' in st.session_state:
            faostat_service = st.session_state.faostat_service
            if hasattr(faostat_service, '_cache'):
                faostat_service._cache.clear()
                faostat_service._cache_timestamps.clear()
        
        # Clear OpenAI cache
        if 'openai_service' in st.session_state:
            openai_service = st.session_state.openai_service
            if hasattr(openai_service, 'clear_cache'):
                openai_service.clear_cache()
        
        st.success("Cache cleared!")

def _show_quick_stats_fast():
    """Show quick statistics with fast loading - prioritize speed over accuracy."""
    
    st.subheader("📈 Quick Stats")
    
    try:
        # Dataset count - use cache if available, otherwise estimate
        dataset_count = "200+"
        if 'faostat_service' in st.session_state:
            faostat_service = st.session_state.faostat_service
            # Only use if already cached (don't trigger new request)
            if hasattr(faostat_service, '_cache') and 'available_datasets' in faostat_service._cache:
                cached_df = faostat_service._cache['available_datasets']
                if cached_df is not None and not cached_df.empty:
                    dataset_count = len(cached_df)
        
        st.metric("📊 Datasets", dataset_count)
        
        # Template count (this is always fast)
        template_count = len(THEMATIC_TEMPLATES)
        if 'template_manager' in st.session_state:
            try:
                templates = st.session_state.template_manager.get_all_templates()
                template_count = len(templates)
            except:
                pass  # Use default
        
        st.metric("📋 Templates", template_count)
        
        # Countries - use fixed estimate for speed
        st.metric("🌍 Countries", "200+")
        
    except Exception as e:
        logger.error(f"Error loading quick stats: {str(e)}")
        # Fallback metrics
        st.metric("📊 Datasets", "200+")
        st.metric("📋 Templates", "4")
        st.metric("🌍 Countries", "200+")

def _show_current_dataset_info():
    """Show information about the currently selected dataset."""
    
    if 'current_dataset_code' in st.session_state and st.session_state.current_dataset_code:
        st.subheader("📊 Current Dataset")
        
        dataset_code = st.session_state.current_dataset_code
        dataset_name = st.session_state.get('current_dataset_name', dataset_code)
        
        st.info(f"**{dataset_name}**\n`{dataset_code}`")
        
        # Show basic dataset info if available
        if 'current_dataset_df' in st.session_state and st.session_state.current_dataset_df is not None:
            df = st.session_state.current_dataset_df
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Cols", len(df.columns))
            
            # Show year range if available
            if 'Year' in df.columns:
                try:
                    year_min = int(df['Year'].min())
                    year_max = int(df['Year'].max())
                    st.caption(f"📅 {year_min} - {year_max}")
                except:
                    pass
        
        # Clear dataset button
        if st.button("🗑️ Clear Dataset", use_container_width=True):
            st.session_state.current_dataset_code = None
            st.session_state.current_dataset_name = None
            st.session_state.current_dataset_df = None
            st.rerun()

def _show_system_status():
    """Show system status and health."""
    
    st.subheader("🔧 System Status")
    
    # Service health checks
    services = []
    
    # FAOSTAT Service
    if 'faostat_service' in st.session_state:
        services.append(("FAOSTAT API", "✅", "green"))
    else:
        services.append(("FAOSTAT API", "❌", "red"))
    
    # OpenAI Service
    if st.session_state.get('openai_service'):
        services.append(("AI Features", "✅", "green"))
    else:
        services.append(("AI Features", "⚠️", "orange"))
    
    # Display status compactly
    for service, status, color in services:
        if color == "green":
            st.success(f"{status} {service}")
        elif color == "orange":
            st.warning(f"{status} {service}")
        else:
            st.error(f"{status} {service}")
    
    # Show current dataset status
    if st.session_state.get('current_dataset_code'):
        st.info(f"📊 Dataset: {st.session_state.current_dataset_code}")