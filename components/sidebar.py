"""
Sidebar component for the FAOSTAT Analytics application.

This module provides the sidebar navigation and configuration interface
for the application, including API key management and page navigation.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional

from config.constants import THEMATIC_TEMPLATES

logger = logging.getLogger(__name__)

def render_sidebar():
    """Render the main sidebar with navigation and configuration."""
    
    with st.sidebar:
        # Application logo/header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2>🌾 FAOSTAT Analytics</h2>
            <p style="font-size: 0.8rem; color: #666;">
                Professional Agricultural Data Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        render_navigation()
        
        st.markdown("---")
        
        # Configuration section
        render_configuration()
        
        st.markdown("---")
        
        # Quick actions
        render_quick_actions()
        
        st.markdown("---")
        
        # Status information
        render_status_info()

def render_navigation():
    """Render the navigation menu."""
    st.markdown("### 📋 Navigation")
    
    pages = [
        ("Home", "🏠"),
        ("Dataset Browser", "📊"),
        ("Analysis", "🔬"),
        ("Thematic Templates", "📋"),
        ("Natural Language Query", "💬"),
        ("About", "ℹ️")
    ]
    
    # Current page selection
    current_page = st.session_state.get('selected_page', 'Home')
    
    for page_name, icon in pages:
        if st.button(f"{icon} {page_name}", key=f"nav_{page_name}"):
            st.session_state.selected_page = page_name
            st.rerun()
    
    # Highlight current page
    if current_page:
        st.success(f"Current: {current_page}")

def render_configuration():
    """Render the configuration section."""
    st.markdown("### ⚙️ Configuration")
    
    # OpenAI API Key configuration
    with st.expander("🤖 AI Configuration", expanded=False):
        render_openai_config()
    
    # Output directory configuration
    with st.expander("📁 Output Settings", expanded=False):
        render_output_config()
    
    # Visualization settings
    with st.expander("📈 Visualization Settings", expanded=False):
        render_viz_config()

def render_openai_config():
    """Render OpenAI configuration interface."""
    config_manager = st.session_state.config_manager
    
    # Current API key status
    current_key = config_manager.get_openai_api_key()
    
    if current_key:
        st.success("✅ API Key configured")
        masked_key = f"sk-...{current_key[-4:]}" if len(current_key) > 4 else "***"
        st.text(f"Current: {masked_key}")
        
        if st.button("🗑️ Remove API Key"):
            config_manager.set_openai_api_key("")
            config_manager.save()
            st.session_state.openai_service = None
            st.success("API key removed")
            st.rerun()
    else:
        st.warning("⚠️ No API Key configured")
    
    # API Key input
    new_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable AI-powered features"
    )
    
    if st.button("💾 Save API Key") and new_api_key:
        if new_api_key.startswith("sk-") and len(new_api_key) > 20:
            config_manager.set_openai_api_key(new_api_key)
            config_manager.save()
            
            # Reinitialize OpenAI service
            from services.openai_service import OpenAIService
            model_config = config_manager.get_model_config()
            st.session_state.openai_service = OpenAIService(
                api_key=new_api_key,
                model=model_config['model'],
                max_tokens=model_config['max_tokens'],
                temperature=model_config['temperature']
            )
            
            st.success("✅ API key saved successfully!")
            st.rerun()
        else:
            st.error("Invalid API key format")
    
    # Model selection
    if current_key:
        model_type = st.selectbox(
            "Model Type",
            options=["comprehensive", "standard"],
            index=0,
            help="Comprehensive: GPT-4 (higher quality, slower)\nStandard: GPT-3.5-turbo (faster, lower cost)"
        )
        
        # Temperature setting
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower values make output more focused, higher values more creative"
        )
        
        # Token usage display
        if st.session_state.openai_service:
            usage = st.session_state.openai_service.get_token_usage_stats()
            if usage['total_tokens'] > 0:
                st.markdown("**Token Usage:**")
                st.text(f"Total: {usage['total_tokens']:,}")
                st.text(f"Prompt: {usage['prompt_tokens']:,}")
                st.text(f"Completion: {usage['completion_tokens']:,}")
                
                # Cost estimation
                cost_info = st.session_state.openai_service.estimate_cost()
                st.text(f"Est. Cost: ${cost_info['total_cost']:.4f}")

def render_output_config():
    """Render output configuration interface."""
    config_manager = st.session_state.config_manager
    
    current_dir = config_manager.get_output_directory()
    st.text(f"Current: {current_dir}")
    
    # Document format selection
    available_formats = []
    if hasattr(st.session_state, 'doc_service'):
        available_formats = st.session_state.doc_service.get_available_formats()
    
    if available_formats:
        default_format = st.selectbox(
            "Default Report Format",
            options=available_formats,
            index=0 if 'pdf' in available_formats else 0
        )
    else:
        st.warning("No document generation libraries available")

def render_viz_config():
    """Render visualization configuration interface."""
    config_manager = st.session_state.config_manager
    
    # Figure quality settings
    dpi = st.slider(
        "Figure DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=50,
        help="Higher DPI = better quality but larger files"
    )
    
    # Figure size
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width (inches)", value=10.0, min_value=4.0, max_value=20.0)
    with col2:
        height = st.number_input("Height (inches)", value=6.0, min_value=3.0, max_value=15.0)
    
    # Color theme
    theme = st.selectbox(
        "Color Theme",
        options=["default", "colorblind", "monochrome"],
        help="Choose visualization color scheme"
    )

def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### ⚡ Quick Actions")
    
    # Dataset refresh
    if st.button("🔄 Refresh Datasets"):
        if hasattr(st.session_state, 'faostat_service'):
            with st.spinner("Refreshing dataset list..."):
                st.session_state.faostat_service.get_available_datasets(force_refresh=True)
            st.success("Datasets refreshed!")
    
    # Clear cache
    if st.button("🗑️ Clear Cache"):
        if hasattr(st.session_state, 'faostat_service'):
            st.session_state.faostat_service._cache.clear()
        if hasattr(st.session_state, 'openai_service') and st.session_state.openai_service:
            st.session_state.openai_service.clear_cache()
        st.success("Cache cleared!")
    
    # Export configuration
    if st.button("💾 Export Config"):
        config_manager = st.session_state.config_manager
        if config_manager.save():
            st.success("Configuration exported!")
        else:
            st.error("Failed to export configuration")

def render_status_info():
    """Render system status information."""
    st.markdown("### 📊 System Status")
    
    # Service status
    services_status = {
        "FAOSTAT API": "✅" if hasattr(st.session_state, 'faostat_service') else "❌",
        "OpenAI API": "✅" if st.session_state.get('openai_service') else "❌",
        "PDF Generation": "✅" if hasattr(st.session_state, 'pdf_service') and st.session_state.pdf_service.is_available() else "❌",
        "Visualizations": "✅" if hasattr(st.session_state, 'viz_generator') else "❌"
    }
    
    for service, status in services_status.items():
        st.text(f"{status} {service}")
    
    # Current dataset info
    if st.session_state.get('current_dataset_code'):
        st.markdown("**Current Dataset:**")
        st.text(st.session_state.current_dataset_code)
    
    # Memory usage (if available)
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        st.text(f"Memory: {memory_percent:.1f}%")
    except ImportError:
        pass

def render_bookmarks_section():
    """Render bookmarks management section."""
    st.markdown("### 🔖 Bookmarks")
    
    bookmarks = st.session_state.get('bookmarks', [])
    
    if bookmarks:
        for i, bookmark in enumerate(bookmarks):
            with st.expander(f"📊 {bookmark.name}"):
                st.text(f"Dataset: {bookmark.dataset_name}")
                st.text(f"Created: {bookmark.created_at}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load", key=f"load_bookmark_{i}"):
                        # Load bookmark logic would go here
                        st.success(f"Loaded: {bookmark.name}")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_bookmark_{i}"):
                        st.session_state.bookmarks.pop(i)
                        st.rerun()
    else:
        st.info("No bookmarks saved yet")
    
    # Add new bookmark
    if st.session_state.get('current_dataset_code'):
        bookmark_name = st.text_input("Bookmark Name", placeholder="My Analysis")
        if st.button("💾 Save Bookmark") and bookmark_name:
            from models.data_models import Bookmark
            from datetime import datetime
            
            new_bookmark = Bookmark(
                id=f"bookmark_{len(bookmarks)}",
                name=bookmark_name,
                dataset_code=st.session_state.current_dataset_code,
                dataset_name=st.session_state.get('current_dataset', {}).get('name', 'Unknown'),
                filters=None,  # Would store current filters
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M')
            )
            
            st.session_state.bookmarks.append(new_bookmark)
            st.success(f"Bookmark '{bookmark_name}' saved!")
            st.rerun()

def render_thematic_templates_quick():
    """Render quick access to thematic templates."""
    st.markdown("### 📋 Quick Templates")
    
    template_options = list(THEMATIC_TEMPLATES.keys())
    
    selected_template = st.selectbox(
        "Select Template",
        options=[""] + template_options,
        format_func=lambda x: THEMATIC_TEMPLATES[x]['name'] if x else "Choose template..."
    )
    
    if selected_template and st.button("🚀 Apply Template"):
        st.session_state.selected_template = selected_template
        st.session_state.selected_page = "Thematic Templates"
        st.rerun()