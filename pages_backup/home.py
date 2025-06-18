"""
Home page for FAOSTAT Analytics Application.

This module provides the main landing page with overview, quick actions,
and getting started information with fast-loading metrics.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def show_home_page():
    """Display the home page with overview and quick actions."""
    
    # Welcome header
    st.markdown("""
    # 🌾 Welcome to FAOSTAT Analytics
    
    **Professional Agricultural Data Analysis Platform**
    
    Unlock insights from FAO's comprehensive agricultural database with AI-powered analytics,
    natural language queries, and professional visualization tools.
    """)
    
    # Fast-loading metrics row
    show_fast_metrics()
    
    st.markdown("---")
    
    # Main content sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_getting_started()
        show_recent_activity()
    
    with col2:
        show_quick_actions()
        show_system_status()

def show_fast_metrics():
    """Show metrics with fast loading - use cached data when possible."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize with reasonable defaults
    dataset_count = "200+"
    country_count = "200+"  # Default to avoid slow loading
    template_count = 4
    years_span = "1961-2023"
    
    # Try to get real dataset count quickly (using cache if available)
    try:
        if 'faostat_service' in st.session_state:
            faostat_service = st.session_state.faostat_service
            
            # Check if we have cached dataset info
            if hasattr(faostat_service, '_cache') and 'available_datasets' in faostat_service._cache:
                # Use cached data - much faster!
                datasets_df = faostat_service._cache['available_datasets']
                if datasets_df is not None and not datasets_df.empty:
                    dataset_count = len(datasets_df)
                    logger.info(f"Using cached dataset count: {dataset_count}")
            else:
                # Try to load datasets quickly with a short timeout
                try:
                    # Set a flag to indicate we're doing a quick check
                    datasets_df = faostat_service.get_available_datasets()
                    if datasets_df is not None and not datasets_df.empty:
                        dataset_count = len(datasets_df)
                        logger.info(f"Got real dataset count: {dataset_count}")
                except Exception as e:
                    logger.warning(f"Could not get real dataset count quickly: {str(e)}")
                    # Keep default value
        
        # Get template count (this is fast)
        if 'template_manager' in st.session_state:
            templates = st.session_state.template_manager.get_all_templates()
            template_count = len(templates)
    
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        # Use fallback values
    
    # Display metrics
    with col1:
        st.metric("📋 Templates", template_count)
    
    with col2:
        st.metric("📊 Datasets", dataset_count)
    
    with col3:
        st.metric("🌍 Countries", country_count)
    
    with col4:
        st.metric("📅 Years", years_span)

def show_getting_started():
    """Show getting started guide."""
    
    st.subheader("🚀 Getting Started")
    
    steps = [
        {
            "title": "1. Browse Datasets",
            "description": "Explore available FAOSTAT datasets covering crops, livestock, trade, and more.",
            "action": "dataset_browser",
            "icon": "📊"
        },
        {
            "title": "2. Configure AI Features",
            "description": "Add your OpenAI API key to enable natural language queries and AI insights.",
            "action": "config",
            "icon": "🤖"
        },
        {
            "title": "3. Start Analysis",
            "description": "Use thematic templates or create custom analysis with our tools.",
            "action": "analysis",
            "icon": "🔬"
        },
        {
            "title": "4. Ask Questions",
            "description": "Query your data in natural language and get instant insights.",
            "action": "nl_query",
            "icon": "💬"
        }
    ]
    
    for step in steps:
        with st.expander(f"{step['icon']} {step['title']}", expanded=False):
            st.write(step["description"])
            
            if step["action"] == "config":
                # Show API key status
                if st.session_state.get('openai_service'):
                    st.success("✅ OpenAI API configured")
                else:
                    st.warning("⚠️ OpenAI API key needed")
                    st.write("Add your API key in the sidebar settings to unlock AI features.")
            
            elif step["action"] == "dataset_browser":
                if st.button("🔍 Browse Datasets", key=f"btn_{step['action']}"):
                    st.session_state.current_page = "dataset_browser"
                    st.rerun()
            
            elif step["action"] == "analysis":
                if st.button("🔬 Start Analysis", key=f"btn_{step['action']}"):
                    st.session_state.current_page = "analysis"
                    st.rerun()
            
            elif step["action"] == "nl_query":
                if st.button("💬 Try Natural Language", key=f"btn_{step['action']}"):
                    st.session_state.current_page = "nl_query"
                    st.rerun()

def show_recent_activity():
    """Show recent activity and bookmarks."""
    
    st.subheader("📝 Recent Activity")
    
    # Check for current dataset
    if st.session_state.get('current_dataset_code'):
        st.info(f"📊 **Current Dataset:** {st.session_state.get('current_dataset_name', st.session_state.current_dataset_code)}")
    
    # Check for recent bookmarks
    bookmarks_found = False
    if 'template_manager' in st.session_state:
        try:
            bookmarks = st.session_state.template_manager.get_all_bookmarks()
            
            if bookmarks:
                st.write("**Recent Bookmarks:**")
                # Show last 3 bookmarks
                recent_bookmarks = list(bookmarks.items())[-3:]
                
                for bookmark_id, bookmark in recent_bookmarks:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📊 **{bookmark.name}**")
                            st.caption(f"Dataset: {bookmark.dataset_name}")
                        with col2:
                            if st.button("Load", key=f"load_recent_{bookmark_id}"):
                                load_bookmark(bookmark)
                bookmarks_found = True
        except Exception as e:
            logger.error(f"Error loading bookmarks: {str(e)}")
    
    # Show analysis results if available
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.write("**Recent Analysis:**")
        results = st.session_state.analysis_results
        st.write(f"📊 **{results.dataset_name}**")
        st.caption(f"Generated: {results.generated_at.strftime('%Y-%m-%d %H:%M')}")
    
    if not bookmarks_found and not st.session_state.get('current_dataset_code'):
        st.info("No recent activity yet. Start by exploring datasets or creating bookmarks!")

def show_quick_actions():
    """Show quick action buttons with real functionality."""
    
    st.subheader("⚡ Quick Actions")
    
    # Load sample dataset with real functionality
    if st.button("📊 Load Sample Dataset", use_container_width=True):
        load_sample_dataset()
    
    # Quick template access
    if st.button("📋 Food Security Template", use_container_width=True):
        st.session_state.selected_template = "food_security"
        st.session_state.current_page = "thematic_templates"
        st.rerun()
    
    # Quick query
    if st.button("💬 Try Sample Query", use_container_width=True):
        if st.session_state.get('openai_service'):
            st.session_state.sample_query = "What are the top 5 rice producing countries?"
            st.session_state.current_page = "nl_query"
            st.rerun()
        else:
            st.warning("Natural language engine not available. Please configure OpenAI API key.")
    
    # Documentation
    if st.button("📚 View Documentation", use_container_width=True):
        st.session_state.current_page = "about"
        st.rerun()

def load_sample_dataset():
    """Load a sample dataset for quick start."""
    
    try:
        if 'faostat_service' in st.session_state:
            faostat_service = st.session_state.faostat_service
            
            with st.spinner("Loading sample dataset (QCL - Crops & Livestock)..."):
                # Get a preview first to check if dataset is available
                df, metadata = faostat_service.get_dataset_preview("QCL", max_rows=100)
                
                if df is not None and not df.empty:
                    # Store preview in session state
                    st.session_state.current_dataset_code = "QCL"
                    st.session_state.current_dataset_name = "Crops and Livestock Products"
                    st.session_state.current_dataset_df = df
                    st.session_state.current_dataset_metadata = metadata
                    
                    st.success(f"✅ Sample dataset loaded! ({len(df):,} preview records)")
                    st.info("💡 Go to Analysis page to start exploring the data")
                    
                    # Optionally load more data in background
                    if st.button("Load Full Dataset"):
                        with st.spinner("Loading full dataset..."):
                            full_df, _ = faostat_service.get_dataset("QCL")
                            if full_df is not None:
                                st.session_state.current_dataset_df = full_df
                                st.success(f"✅ Full dataset loaded! ({len(full_df):,} records)")
                else:
                    st.error("Failed to load sample dataset")
        else:
            st.error("FAOSTAT service not available")
            
    except Exception as e:
        logger.error(f"Error loading sample dataset: {str(e)}")
        st.error(f"Error loading sample dataset: {str(e)}")

def load_bookmark(bookmark):
    """Load a bookmark into the current session."""
    
    try:
        # Set dataset information
        st.session_state.current_dataset_code = bookmark.dataset_code
        st.session_state.current_dataset_name = bookmark.dataset_name
        
        # Try to load the actual dataset
        if 'faostat_service' in st.session_state:
            with st.spinner(f"Loading bookmark: {bookmark.name}..."):
                faostat_service = st.session_state.faostat_service
                df, metadata = faostat_service.get_dataset_preview(bookmark.dataset_code, max_rows=100)
                
                if df is not None:
                    st.session_state.current_dataset_df = df
                    st.session_state.current_dataset_metadata = metadata
                    st.success(f"✅ Loaded bookmark: {bookmark.name}")
                else:
                    st.warning(f"Could not load data for bookmark: {bookmark.name}")
        
        # Navigate to analysis
        st.session_state.current_page = "analysis"
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error loading bookmark: {str(e)}")
        st.error(f"Error loading bookmark: {str(e)}")

def show_system_status():
    """Show system status and health."""
    
    st.subheader("🔧 System Status")
    
    # Service health checks with more detail
    services = []
    
    # FAOSTAT Service
    if 'faostat_service' in st.session_state:
        services.append(("FAOSTAT API", "✅", "green", "Connected"))
    else:
        services.append(("FAOSTAT API", "❌", "red", "Not available"))
    
    # OpenAI Service
    if st.session_state.get('openai_service'):
        services.append(("AI Features", "✅", "green", "Ready"))
    else:
        services.append(("AI Features", "⚠️", "orange", "API key needed"))
    
    # Template Manager
    if 'template_manager' in st.session_state:
        templates = st.session_state.template_manager.get_all_templates()
        services.append(("Templates", "✅", "green", f"{len(templates)} available"))
    else:
        services.append(("Templates", "❌", "red", "Not loaded"))
    
    # Display status compactly
    for service, status, color, detail in services:
        if color == "green":
            st.success(f"{status} {service}")
        elif color == "orange":
            st.warning(f"{status} {service}")
        else:
            st.error(f"{status} {service}")
    
    # Performance info (optional)
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            st.warning(f"🧠 Memory: {memory_percent:.1f}%")
        else:
            st.info(f"🧠 Memory: {memory_percent:.1f}%")
    except ImportError:
        pass

if __name__ == "__main__":
    show_home_page()