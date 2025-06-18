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

# Page configuration
st.set_page_config(
    page_title="FAOSTAT Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # FAOSTAT Analytics
        
        Advanced analytics platform for FAOSTAT data with AI-powered insights,
        natural language queries, and comprehensive visualization capabilities.
        
        Built with Streamlit, OpenAI, and modern data science tools.
        """
    }
)

# Import application modules
try:
    from config.config_manager import ConfigManager
    from services.faostat_service import FAOStatService
    from services.openai_service import OpenAIService
    from features.template_manager import TemplateManager
    from features.nl_query_engine import NaturalLanguageQueryEngine
    from utils.visualization import VisualizationGenerator
    from services.document_service import DocumentService
    from components.sidebar import create_sidebar
    
    # Import pages
    from pages_backup.home import show_home_page
    from pages_backup.dataset_browser import show_dataset_browser
    from pages_backup.analysis import show_analysis_page
    from pages_backup.thematic_templates import show_thematic_templates_page
    from pages_backup.nl_query import show_nl_query_page
    from pages_backup.about import show_about_page
    
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

st.write("✅ APP.PY IS RUNNING - OUTSIDE MAIN")
logger.info("✅ APP.PY IS RUNNING - OUTSIDE MAIN")


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
        
        # Initialize visualization generator
        viz_generator = VisualizationGenerator(
            output_dir=os.path.join(config.get_output_directory(), "figures")
        )
        st.session_state.viz_generator = viz_generator
        
        # Initialize document service
        doc_service = DocumentService(config)
        st.session_state.doc_service = doc_service
        
        # Initialize OpenAI service if API key is available
        openai_api_key = config.get_openai_api_key() or st.session_state.get('openai_api_key')
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
        if 'current_dataset_code' not in st.session_state:
            st.session_state.current_dataset_code = None
        
        if 'current_dataset_name' not in st.session_state:
            st.session_state.current_dataset_name = None
        
        if 'current_dataset_df' not in st.session_state:
            st.session_state.current_dataset_df = None
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        st.session_state.services_initialized = True
        logger.info("Services initialized successfully")
        
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        logger.error(f"Service initialization error: {str(e)}")
        st.stop()

def main():
    """Main application function."""
    
    st.write("✅ MAIN() FUNCTION ENTERED")
    logger.info("✅ MAIN() FUNCTION ENTERED")
    
    # Initialize services
    initialize_services()
    
    # Create sidebar navigation (this replaces the old confusing navigation)
    selected_page = create_sidebar()
    
    st.write(f"DEBUG selected_page: '{selected_page}'")
    logger.info(f"DEBUG selected_page: '{selected_page}'")
    # Main content area
    try:
        # Route to selected page based on sidebar navigation
        
        if selected_page == "home":
            logger.info("Displaying Home Page")
            show_home_page()
        
        elif selected_page == "dataset_browser":
            logger.info("Displaying Dataset Browser Page")
            show_dataset_browser()
        
        elif selected_page == "analysis":
            logger.info("Displaying Analysis Page")
            show_analysis_page()
        
        elif selected_page == "thematic_templates":
            logger.info("Displaying Thematic Templates Page")
            show_thematic_templates_page()
        
        elif selected_page == "nl_query":
            logger.info("Displaying Natural Language Query Page")
            show_nl_query_page()
        
        elif selected_page == "about":
            logger.info("Displaying About Page")
            show_about_page()
        
        else:
            st.error(f"Unknown page: {selected_page}")
            logger.warning(f"Unknown page selected: {selected_page}")
    
    except Exception as e:
        st.error(f"Error displaying page '{selected_page}': {str(e)}")
        logger.error(f"Page display error for {selected_page}: {str(e)}")
        
        # Show debug information in development
        if st.checkbox("Show debug information"):
            st.exception(e)

def show_footer():
    """Show application footer."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
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
        # Run main application
        main()
        
        # Show footer
        show_footer()
        
    except Exception as e:
        st.error("Critical application error occurred!")
        st.exception(e)
        logger.critical(f"Critical application error: {str(e)}")