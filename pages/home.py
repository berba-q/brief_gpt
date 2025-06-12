"""
Home page for FAOSTAT Analytics Application.

This module provides the main landing page with overview, quick actions,
and getting started information.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

def show_home_page():
    """Display the home page with overview and quick actions."""
    
    # Welcome header
    st.markdown("""
    # 🌾 Welcome to FAOSTAT Analytics
    
    **Professional Agricultural Data Analysis Platform**
    
    Unlock insights from FAO's comprehensive agricultural database with AI-powered analytics,
    natural language queries, and professional visualization tools.
    """)
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Template count
        template_count = 0
        if 'template_manager' in st.session_state:
            templates = st.session_state.template_manager.get_all_templates()
            template_count = len(templates)
        st.metric("📋 Templates", template_count)
    
    with col2:
        # Available datasets (placeholder)
        st.metric("📊 Datasets", "200+")
    
    with col3:
        # Countries covered
        st.metric("🌍 Countries", "245+")
    
    with col4:
        # Years of data
        st.metric("📅 Years", "1960-2023")
    
    st.markdown("---")
    
    # Main content sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_getting_started()
        show_recent_activity()
    
    with col2:
        show_quick_actions()
        show_system_status()

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
                if 'config' in st.session_state:
                    api_key = st.session_state.config.get_openai_api_key()
                    if api_key:
                        st.success("✅ OpenAI API configured")
                    else:
                        st.warning("⚠️ OpenAI API key needed")
                        st.write("Add your API key in the sidebar settings to unlock AI features.")
            
            elif step["action"] == "dataset_browser":
                if st.button("🔍 Browse Datasets", key=f"btn_{step['action']}"):
                    st.session_state.selected_page = "dataset_browser"
                    st.rerun()
            
            elif step["action"] == "analysis":
                if st.button("🔬 Start Analysis", key=f"btn_{step['action']}"):
                    st.session_state.selected_page = "analysis"
                    st.rerun()
            
            elif step["action"] == "nl_query":
                if st.button("💬 Try Natural Language", key=f"btn_{step['action']}"):
                    st.session_state.selected_page = "nl_query"
                    st.rerun()

def show_recent_activity():
    """Show recent activity and bookmarks."""
    
    st.subheader("📝 Recent Activity")
    
    # Check for recent bookmarks
    if 'template_manager' in st.session_state:
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
                        st.caption(f"Dataset: {bookmark.dataset_name} | Created: {bookmark.created_at}")
                    with col2:
                        if st.button("Load", key=f"load_recent_{bookmark_id}"):
                            st.info(f"Loading bookmark: {bookmark.name}")
        else:
            st.info("No recent activity yet. Start by exploring datasets or creating bookmarks!")
    else:
        st.info("Template manager not initialized")
    
    # Show query history if available
    if 'query_history' in st.session_state and st.session_state.query_history:
        st.write("**Recent Queries:**")
        recent_queries = st.session_state.query_history[-3:]
        
        for i, query in enumerate(recent_queries):
            st.write(f"💬 {query.get('query', 'Unknown query')}")
            st.caption(f"Dataset: {query.get('dataset', 'Unknown')}")

def show_quick_actions():
    """Show quick action buttons."""
    
    st.subheader("⚡ Quick Actions")
    
    # Quick dataset access
    if st.button("📊 Load Sample Dataset", use_container_width=True):
        with st.spinner("Loading QCL (Crops & Livestock) dataset..."):
            try:
                if 'faostat_service' in st.session_state:
                    df, metadata = st.session_state.faostat_service.get_dataset("QCL")
                    if df is not None:
                        st.session_state.current_dataset = "QCL"
                        st.session_state.current_dataset_df = df.head(1000)  # Limit for demo
                        st.success("✅ Sample dataset loaded!")
                        st.rerun()
                    else:
                        st.error("Failed to load dataset")
                else:
                    st.error("FAOSTAT service not available")
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    # Quick template access
    if st.button("📋 Food Security Template", use_container_width=True):
        st.session_state.selected_template = "food_security"
        st.session_state.selected_page = "thematic_templates"
        st.rerun()
    
    # Quick query
    if st.button("💬 Try Sample Query", use_container_width=True):
        if 'nl_engine' in st.session_state:
            st.session_state.sample_query = "What are the top 5 rice producing countries?"
            st.session_state.selected_page = "nl_query"
            st.rerun()
        else:
            st.warning("Natural language engine not available. Please configure OpenAI API key.")
    
    # Documentation
    if st.button("📚 View Documentation", use_container_width=True):
        st.session_state.selected_page = "about"
        st.rerun()

def show_system_status():
    """Show system status and health."""
    
    st.subheader("🔧 System Status")
    
    # Service health checks
    services = {}
    
    # FAOSTAT Service
    if 'faostat_service' in st.session_state:
        services["FAOSTAT API"] = {"status": "✅", "color": "green"}
    else:
        services["FAOSTAT API"] = {"status": "❌", "color": "red"}
    
    # OpenAI Service
    if 'openai_service' in st.session_state and st.session_state.openai_service:
        services["OpenAI API"] = {"status": "✅", "color": "green"}
    else:
        services["OpenAI API"] = {"status": "⚠️", "color": "orange"}
    
    # Template Manager
    if 'template_manager' in st.session_state:
        services["Templates"] = {"status": "✅", "color": "green"}
    else:
        services["Templates"] = {"status": "❌", "color": "red"}
    
    # NL Query Engine
    if 'nl_engine' in st.session_state:
        services["NL Queries"] = {"status": "✅", "color": "green"}
    else:
        services["NL Queries"] = {"status": "⚠️", "color": "orange"}
    
    # Display status
    for service, info in services.items():
        if info["color"] == "green":
            st.success(f"{info['status']} {service}")
        elif info["color"] == "orange":
            st.warning(f"{info['status']} {service}")
        else:
            st.error(f"{info['status']} {service}")
    
    # Show current dataset if any
    if 'current_dataset' in st.session_state and st.session_state.current_dataset:
        st.info(f"📊 Current Dataset: {st.session_state.current_dataset}")
    
    # Show memory usage if available
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            st.warning(f"🧠 Memory: {memory_percent:.1f}%")
        else:
            st.success(f"🧠 Memory: {memory_percent:.1f}%")
    except ImportError:
        pass

def show_feature_highlights():
    """Show feature highlights and capabilities."""
    
    st.subheader("✨ Key Features")
    
    features = [
        {
            "title": "AI-Powered Analysis",
            "description": "Generate insights automatically using GPT-4 with your agricultural data.",
            "icon": "🤖"
        },
        {
            "title": "Natural Language Queries",
            "description": "Ask questions in plain English and get data-driven answers instantly.",
            "icon": "💬"
        },
        {
            "title": "Thematic Templates",
            "description": "Pre-built analysis frameworks for food security, climate impact, and more.",
            "icon": "📋"
        },
        {
            "title": "Professional Visualizations",
            "description": "Generate publication-ready charts, maps, and interactive visualizations.",
            "icon": "📊"
        },
        {
            "title": "Export & Share",
            "description": "Export analysis to PDF reports, Word documents, or share online.",
            "icon": "📄"
        },
        {
            "title": "Bookmark System",
            "description": "Save and organize your analysis configurations for future use.",
            "icon": "🔖"
        }
    ]
    
    # Display in grid
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            **{feature['icon']} {feature['title']}**
            
            {feature['description']}
            """)

def show_data_sources_info():
    """Show information about FAOSTAT data sources."""
    
    st.subheader("📊 Data Sources")
    
    st.markdown("""
    **FAOSTAT** provides free access to food and agriculture data for over 245 countries and territories:
    
    - **Production Data**: Crops, livestock, forestry, and fisheries
    - **Trade Data**: Import/export statistics and trade matrices  
    - **Food Security**: Food balance sheets and utilization accounts
    - **Emissions**: Agricultural greenhouse gas emissions
    - **Land Use**: Agricultural land and land use change
    - **Prices**: Producer and consumer price indices
    
    Data coverage spans from **1961 to present** with regular updates.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌍 Countries", "245+")
    with col2:
        st.metric("📅 Years", "60+")
    with col3:
        st.metric("📊 Indicators", "20,000+")
    
    st.info("💡 **Tip**: Start with the Dataset Browser to explore available data categories.")

if __name__ == "__main__":
    show_home_page()