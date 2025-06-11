"""
Home page for the FAOSTAT Analytics application.

This module provides the main landing page with overview information,
recent activity, and quick access to key features.
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from config.constants import THEMATIC_TEMPLATES

logger = logging.getLogger(__name__)

def render():
    """Render the home page."""
    
    # Welcome section
    render_welcome_section()
    
    # Quick start section
    render_quick_start_section()
    
    # Recent activity
    render_recent_activity()
    
    # System overview
    render_system_overview()
    
    # Featured templates
    render_featured_templates()

def render_welcome_section():
    """Render the welcome section with overview."""
    
    st.markdown("""
    ## Welcome to FAOSTAT Analytics! 🌾
    
    Your comprehensive platform for analyzing Food and Agriculture Organization (FAO) statistical data 
    and generating professional analytical briefs with AI-powered insights.
    """)
    
    # Key features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Data Analysis
        - Access 200+ FAOSTAT datasets
        - Interactive filtering and exploration
        - Advanced statistical analysis
        - Professional visualizations
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Insights
        - GPT-powered analysis
        - Automated brief generation
        - Natural language queries
        - Expert-level interpretations
        """)
    
    with col3:
        st.markdown("""
        ### 📄 Report Generation
        - PDF and Word documents
        - FAO-standard formatting
        - Publication-ready output
        - Customizable templates
        """)

def render_quick_start_section():
    """Render quick start options."""
    
    st.markdown("## 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### New to FAOSTAT Analytics?")
        
        if st.button("🎯 Take the Tour", use_container_width=True):
            st.session_state.selected_page = "About"
            st.rerun()
        
        if st.button("📖 Browse Datasets", use_container_width=True):
            st.session_state.selected_page = "Dataset Browser"
            st.rerun()
        
        if st.button("🔬 Start Analysis", use_container_width=True):
            st.session_state.selected_page = "Analysis"
            st.rerun()
    
    with col2:
        st.markdown("### Ready to analyze?")
        
        if st.button("📋 Use Template", use_container_width=True):
            st.session_state.selected_page = "Thematic Templates"
            st.rerun()
        
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.selected_page = "Natural Language Query"
            st.rerun()
        
        # API key status
        openai_service = st.session_state.get('openai_service')
        if openai_service:
            st.success("✅ AI features enabled")
        else:
            st.warning("⚠️ Configure OpenAI API key in sidebar for AI features")

def render_recent_activity():
    """Render recent activity section."""
    
    st.markdown("## 📈 Recent Activity")
    
    # Check if we have any recent analysis
    analysis_results = st.session_state.get('analysis_results')
    current_dataset = st.session_state.get('current_dataset_code')
    
    if analysis_results or current_dataset:
        col1, col2 = st.columns(2)
        
        with col1:
            if current_dataset:
                st.markdown("### 📊 Current Dataset")
                st.info(f"Working with: **{current_dataset}**")
                
                if st.button("🔄 Continue Analysis"):
                    st.session_state.selected_page = "Analysis"
                    st.rerun()
        
        with col2:
            if analysis_results:
                st.markdown("### 📋 Last Analysis")
                st.success(f"Completed: **{analysis_results.dataset_name}**")
                st.text(f"Generated: {analysis_results.generated_at.strftime('%Y-%m-%d %H:%M')}")
                
                if st.button("📄 View Results"):
                    st.session_state.selected_page = "Analysis"
                    st.rerun()
    else:
        st.info("No recent activity. Start by browsing datasets or selecting a template!")

def render_system_overview():
    """Render system status and capabilities overview."""
    
    st.markdown("## 🔧 System Status")
    
    # Service status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        faostat_status = "✅ Connected" if hasattr(st.session_state, 'faostat_service') else "❌ Unavailable"
        st.metric("FAOSTAT API", faostat_status)
    
    with col2:
        ai_status = "✅ Enabled" if st.session_state.get('openai_service') else "❌ Disabled"
        st.metric("AI Features", ai_status)
    
    with col3:
        pdf_status = "✅ Available" if (hasattr(st.session_state, 'pdf_service') and 
                                      st.session_state.pdf_service.is_available()) else "❌ Unavailable"
        st.metric("PDF Generation", pdf_status)
    
    with col4:
        viz_status = "✅ Ready" if hasattr(st.session_state, 'viz_generator') else "❌ Not Ready"
        st.metric("Visualizations", viz_status)
    
    # Dataset statistics
    if hasattr(st.session_state, 'faostat_service'):
        with st.expander("📊 Available Datasets", expanded=False):
            try:
                datasets_df = st.session_state.faostat_service.get_available_datasets()
                
                if not datasets_df.empty:
                    st.success(f"✅ {len(datasets_df)} datasets available")
                    
                    # Show some statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Datasets", len(datasets_df))
                    with col2:
                        # Count recently updated datasets (last 2 years)
                        current_year = datetime.now().year
                        recent_datasets = datasets_df[
                            datasets_df['last_updated'].str.contains(str(current_year-1) + '|' + str(current_year), na=False)
                        ]
                        st.metric("Recently Updated", len(recent_datasets))
                    
                    # Show top 5 datasets by name
                    st.markdown("**Popular Datasets:**")
                    popular_datasets = [
                        "Crops and livestock products",
                        "Food Balances",
                        "Fertilizers by Nutrient",
                        "Trade: Crops and Livestock Products",
                        "Population"
                    ]
                    
                    available_popular = datasets_df[
                        datasets_df['name'].str.contains('|'.join(popular_datasets), case=False, na=False)
                    ]['name'].head(5).tolist()
                    
                    for dataset in available_popular:
                        st.text(f"• {dataset}")
                        
                else:
                    st.warning("⚠️ No datasets loaded yet")
                    if st.button("🔄 Load Datasets"):
                        with st.spinner("Loading datasets..."):
                            datasets_df = st.session_state.faostat_service.get_available_datasets(force_refresh=True)
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading dataset information: {str(e)}")
                logger.error(f"Error in home page dataset loading: {str(e)}")

def render_featured_templates():
    """Render featured thematic templates."""
    
    st.markdown("## 📋 Featured Analysis Templates")
    
    st.markdown("""
    Get started quickly with our pre-configured analysis templates designed for common agricultural policy questions.
    """)
    
    # Display featured templates
    featured_templates = [
        "food_security",
        "climate_impact", 
        "fertilizer_analysis"
    ]
    
    cols = st.columns(len(featured_templates))
    
    for i, template_key in enumerate(featured_templates):
        if template_key in THEMATIC_TEMPLATES:
            template = THEMATIC_TEMPLATES[template_key]
            
            with cols[i]:
                with st.container():
                    st.markdown(f"### {template['name']}")
                    st.markdown(f"_{template['description']}_")
                    
                    # Show key datasets
                    st.markdown("**Datasets:**")
                    for dataset in template['datasets'][:2]:  # Show first 2
                        st.text(f"• {dataset['name']}")
                    if len(template['datasets']) > 2:
                        st.text(f"• ... and {len(template['datasets']) - 2} more")
                    
                    # Show focus regions
                    if template.get('regions_focus'):
                        st.markdown("**Focus Areas:**")
                        regions_text = ", ".join(template['regions_focus'][:2])
                        if len(template['regions_focus']) > 2:
                            regions_text += f" +{len(template['regions_focus']) - 2} more"
                        st.text(regions_text)
                    
                    if st.button(f"🚀 Use {template['name']}", key=f"template_{template_key}"):
                        st.session_state.selected_template = template_key
                        st.session_state.selected_page = "Thematic Templates"
                        st.rerun()

def render_tips_and_tricks():
    """Render tips and tricks section."""
    
    st.markdown("## 💡 Tips & Tricks")
    
    tips = [
        {
            "title": "🔍 Smart Dataset Search",
            "content": "Use keywords like 'production', 'trade', or 'population' to quickly find relevant datasets."
        },
        {
            "title": "🎯 Effective Filtering", 
            "content": "Start with recent years (2010+) and major regions for faster processing and clearer trends."
        },
        {
            "title": "📊 Better Visualizations",
            "content": "Limit to top 10-15 countries/items for readable charts. Use time series for trend analysis."
        },
        {
            "title": "🤖 AI Query Tips",
            "content": "Ask specific questions like 'What are the trends in wheat production in Asia?' for better AI responses."
        },
        {
            "title": "📄 Professional Reports",
            "content": "Use thematic templates for publication-ready briefs that follow FAO formatting standards."
        }
    ]
    
    # Display tips in expandable sections
    for tip in tips:
        with st.expander(tip["title"]):
            st.markdown(tip["content"])

def render_news_and_updates():
    """Render news and updates section."""
    
    st.markdown("## 📢 What's New")
    
    # Sample updates - in real implementation, these could come from a config file or API
    updates = [
        {
            "date": "2024-12-15",
            "title": "New AI-Powered Natural Language Queries",
            "description": "Ask questions about your data in plain English and get instant insights.",
            "type": "feature"
        },
        {
            "date": "2024-12-10", 
            "title": "Enhanced PDF Report Generation",
            "description": "Professional reports now include improved formatting and FAO branding.",
            "type": "improvement"
        },
        {
            "date": "2024-12-05",
            "title": "Updated FAOSTAT API Integration",
            "description": "Faster data loading and improved error handling for better reliability.",
            "type": "update"
        }
    ]
    
    for update in updates:
        # Color coding by type
        if update["type"] == "feature":
            st.success(f"🆕 **{update['title']}** ({update['date']})")
        elif update["type"] == "improvement":
            st.info(f"✨ **{update['title']}** ({update['date']})")
        else:
            st.info(f"🔄 **{update['title']}** ({update['date']})")
        
        st.markdown(f"   {update['description']}")
        st.markdown("")

def render_getting_started_checklist():
    """Render a getting started checklist for new users."""
    
    st.markdown("## ✅ Getting Started Checklist")
    
    checklist_items = [
        {
            "task": "Configure OpenAI API Key",
            "description": "Enable AI-powered insights and natural language queries",
            "check_condition": lambda: st.session_state.get('openai_service') is not None,
            "action_page": None
        },
        {
            "task": "Browse Available Datasets", 
            "description": "Explore the 200+ FAOSTAT datasets available for analysis",
            "check_condition": lambda: st.session_state.get('current_dataset_code') is not None,
            "action_page": "Dataset Browser"
        },
        {
            "task": "Run Your First Analysis",
            "description": "Generate visualizations and insights from agricultural data",
            "check_condition": lambda: st.session_state.get('analysis_results') is not None,
            "action_page": "Analysis"
        },
        {
            "task": "Generate a Professional Brief",
            "description": "Create a publication-ready PDF or Word document",
            "check_condition": lambda: False,  # Could track if user has generated a report
            "action_page": "Thematic Templates"
        }
    ]
    
    for i, item in enumerate(checklist_items):
        is_complete = item["check_condition"]()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            status_icon = "✅" if is_complete else "⏳"
            st.markdown(f"{status_icon} **{item['task']}**")
            st.markdown(f"   _{item['description']}_")
        
        with col2:
            if not is_complete and item["action_page"]:
                if st.button(f"Go →", key=f"checklist_{i}"):
                    st.session_state.selected_page = item["action_page"]
                    st.rerun()
    
    # Progress calculation
    completed_items = sum(1 for item in checklist_items if item["check_condition"]())
    progress = completed_items / len(checklist_items)
    
    st.progress(progress)
    st.markdown(f"**Progress: {completed_items}/{len(checklist_items)} completed ({progress*100:.0f}%)**")

# Main render function with sections
def render():
    """Render the complete home page."""
    
    # Welcome section
    render_welcome_section()
    
    st.markdown("---")
    
    # Quick start section
    render_quick_start_section()
    
    st.markdown("---")
    
    # Getting started checklist for new users
    render_getting_started_checklist()
    
    st.markdown("---")
    
    # Recent activity
    render_recent_activity()
    
    st.markdown("---")
    
    # System overview
    render_system_overview()
    
    st.markdown("---")
    
    # Featured templates
    render_featured_templates()
    
    st.markdown("---")
    
    # Tips and tricks
    render_tips_and_tricks()
    
    st.markdown("---")
    
    # News and updates
    render_news_and_updates()