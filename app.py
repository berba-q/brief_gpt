"""
FAOSTAT Analytics Application

Main Streamlit application for analyzing FAOSTAT data with AI-powered insights,
natural language queries, and comprehensive visualization capabilities.
"""

import streamlit as st
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import application modules
try:
    from config.config_manager import ConfigManager
    from services.faostat_service import FAOStatService
    from services.openai_service import OpenAIService
    from features.template_manager import TemplateManager
    from features.nl_query_engine import NaturalLanguageQueryEngine
    from components.sidebar import create_sidebar
    
    # Import pages
    from pages.home import show_home_page
    from pages.dataset_browser import show_dataset_browser
    from pages.analysis import show_analysis_page
    from pages.thematic_templates import show_thematic_templates_page
    from pages.nl_query import show_nl_query_page
    from pages.about import show_about_page
    
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FAOSTAT Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': '#',
        'Report a bug': '#',
        'About': """
        # FAOSTAT Analytics
        
        Advanced analytics platform for FAOSTAT data with AI-powered insights,
        natural language queries, and comprehensive visualization capabilities.
        
        Built with Streamlit, OpenAI, and modern data science tools.
        """
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-color: #c3e6cb;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def initialize_services():
    """Initialize all application services and store in session state."""
    
    if 'services_initialized' in st.session_state:
        return
    
    try:
        # Initialize configuration manager
        config = ConfigManager()
        st.session_state.config = config
        
        # Initialize FAOSTAT service
        faostat_service = FAOStatService(
            base_url=config.get_faostat_base_url(),
            cache_duration=config.get_int('faostat.cache_duration', 3600)
        )
        st.session_state.faostat_service = faostat_service
        
        # Initialize OpenAI service if API key is available
        openai_api_key = config.get_openai_api_key()
        if openai_api_key:
            model_config = config.get_model_config('comprehensive')
            openai_service = OpenAIService(
                api_key=openai_api_key,
                model=model_config.get('model', 'gpt-4'),
                max_tokens=model_config.get('max_tokens', 1500),
                temperature=model_config.get('temperature', 0.7)
            )
            st.session_state.openai_service = openai_service
            
            # Initialize NL Query Engine
            nl_engine = NaturalLanguageQueryEngine(openai_service, faostat_service)
            st.session_state.nl_engine = nl_engine
        else:
            st.session_state.openai_service = None
            st.session_state.nl_engine = None
        
        # Initialize Template Manager
        template_manager = TemplateManager()
        st.session_state.template_manager = template_manager
        
        # Initialize session state variables
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        
        if 'current_dataset_df' not in st.session_state:
            st.session_state.current_dataset_df = None
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        st.session_state.services_initialized = True
        logger.info("Services initialized successfully")
        
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        logger.error(f"Service initialization error: {str(e)}")
        st.stop()

def main():
    """Main application function."""
    
    # Initialize services
    initialize_services()
    
    # Main header
    st.title("🌾 FAOSTAT Analytics")
    st.markdown("*Advanced analytics platform for FAO agricultural data*")
    
    # Create sidebar navigation
    selected_page = create_sidebar()
    
    # Check for OpenAI configuration
    if not st.session_state.get('openai_service') and selected_page in ['nl_query', 'thematic_templates']:
        st.warning("""
        ⚠️ **OpenAI API Key Required**
        
        Some features require an OpenAI API key to function properly. Please configure your API key in the configuration file or contact your administrator.
        
        Features requiring OpenAI:
        - Natural Language Queries
        - AI-Powered Insights
        - Thematic Analysis Templates
        """)
    
    # Route to selected page
    try:
        if selected_page == "home":
            show_home_page()
        
        elif selected_page == "dataset_browser":
            show_dataset_browser()
        
        elif selected_page == "analysis":
            show_analysis_page()
        
        elif selected_page == "thematic_templates":
            show_thematic_templates_page()
        
        elif selected_page == "nl_query":
            show_nl_query_page()
        
        elif selected_page == "about":
            show_about_page()
        
        else:
            st.error(f"Unknown page: {selected_page}")
    
    except Exception as e:
        st.error(f"Error displaying page '{selected_page}': {str(e)}")
        logger.error(f"Page display error for {selected_page}: {str(e)}")
        
        # Show debug information in development
        if st.checkbox("Show debug information"):
            st.exception(e)

def show_startup_checks():
    """Show system startup checks and status."""
    
    with st.expander("🔍 System Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Services Status:**")
            
            # FAOSTAT Service
            if st.session_state.get('faostat_service'):
                st.success("✅ FAOSTAT Service")
            else:
                st.error("❌ FAOSTAT Service")
            
            # OpenAI Service
            if st.session_state.get('openai_service'):
                st.success("✅ OpenAI Service")
            else:
                st.warning("⚠️ OpenAI Service (API key required)")
            
            # Template Manager
            if st.session_state.get('template_manager'):
                st.success("✅ Template Manager")
            else:
                st.error("❌ Template Manager")
            
            # NL Query Engine
            if st.session_state.get('nl_engine'):
                st.success("✅ NL Query Engine")
            else:
                st.warning("⚠️ NL Query Engine (requires OpenAI)")
        
        with col2:
            st.write("**Configuration:**")
            
            config = st.session_state.get('config')
            if config:
                st.write(f"- Output Directory: {config.get_output_directory()}")
                st.write(f"- FAOSTAT URL: {config.get_faostat_base_url()}")
                
                if config.get_openai_api_key():
                    st.write("- OpenAI API: ✅ Configured")
                else:
                    st.write("- OpenAI API: ❌ Not configured")
            else:
                st.error("Configuration not loaded")

# Add footer
def show_footer():
    """Show application footer."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>
                FAOSTAT Analytics Platform | 
                Data from <a href='https://www.fao.org/faostat/' target='_blank'>FAO Statistics</a> | 
                Powered by <a href='https://streamlit.io' target='_blank'>Streamlit</a> & 
                <a href='https://openai.com' target='_blank'>OpenAI</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        # Show system status for debugging
        if st.sidebar.checkbox("Show System Status", value=False):
            show_startup_checks()
        
        # Run main application
        main()
        
        # Show footer
        show_footer()
        
    except Exception as e:
        st.error("Critical application error occurred!")
        st.exception(e)
        logger.critical(f"Critical application error: {str(e)}")