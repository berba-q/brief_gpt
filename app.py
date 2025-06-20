"""
FAOSTAT Analytical Brief Generator

Streamlined application for generating FAO-style analytical briefs with AI-powered insights.
Focus: Dataset selection -> Configuration -> Brief generation -> Export
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
    page_title="FAOSTAT Brief Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import core modules
try:
    from config.config_manager import ConfigManager
    from services.faostat_service import FAOStatService
    from services.openai_service import OpenAIService
    from services.geography_service import GeographyService
    from services.brief_generator import BriefGeneratorService
    from services.document_service import ExportService
    from utils.visualization import VisualizationGenerator
    
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

def initialize_services():
    """Initialize all application services."""
    if 'services_initialized' in st.session_state:
        return
    
    try:
        # Initialize configuration manager
        config = ConfigManager()
        st.session_state.config = config
        
        # Initialize FAOSTAT service (keep as-is)
        faostat_service = FAOStatService(
            base_url=config.get_faostat_base_url(),
            cache_duration=config.get_int('faostat.cache_duration', 3600)
        )
        st.session_state.faostat_service = faostat_service
        
        # Initialize geography service for area code classification
        geography_service = GeographyService(faostat_service)
        st.session_state.geography_service = geography_service
        
        # Initialize visualization generator
        viz_generator = VisualizationGenerator(
            output_dir=os.path.join(config.get_output_directory(), "figures")
        )
        st.session_state.viz_generator = viz_generator
        
        # Initialize OpenAI service if API key is available
        openai_api_key = config.get_openai_api_key() or st.session_state.get('openai_api_key')
        if openai_api_key:
            openai_service = OpenAIService(
                api_key=openai_api_key,
                model='gpt-4',
                max_tokens=2000,
                temperature=0.7
            )
            st.session_state.openai_service = openai_service
            
            # Initialize Brief Generator
            brief_generator = BriefGeneratorService(
                openai_service=openai_service,
                faostat_service=faostat_service,
                geography_service=geography_service,
                viz_generator=viz_generator
            )
            st.session_state.brief_generator = brief_generator
            
        else:
            st.session_state.openai_service = None
            st.session_state.brief_generator = None
        
        # Initialize Export Service
        export_service = ExportService(config)
        st.session_state.export_service = export_service
        
        # Initialize session state for app flow
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'dataset_selection'
        
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = None
            
        if 'analysis_config' not in st.session_state:
            st.session_state.analysis_config = {}
            
        if 'generated_brief' not in st.session_state:
            st.session_state.generated_brief = None
        
        st.session_state.services_initialized = True
        logger.info("Services initialized successfully")
        
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        logger.error(f"Service initialization error: {str(e)}")
        st.stop()

def create_sidebar():
    """Create sidebar with navigation and current status."""
    with st.sidebar:
        st.title("📊 Brief Generator")
        
        # Progress indicator
        steps = [
            ("📊", "Dataset Selection"),
            ("⚙️", "Configuration"),
            ("📝", "Brief Generation"),
            ("📁", "Export")
        ]
        
        current_step = st.session_state.get('current_step', 'dataset_selection')
        
        st.markdown("### Progress")
        for i, (icon, label) in enumerate(steps):
            step_keys = ['dataset_selection', 'configuration', 'brief_generation', 'export']
            is_current = current_step == step_keys[i]
            is_completed = False
            
            # Check if step is completed
            if step_keys[i] == 'dataset_selection' and st.session_state.get('selected_dataset'):
                is_completed = True
            elif step_keys[i] == 'configuration' and st.session_state.get('analysis_config'):
                is_completed = True
            elif step_keys[i] == 'brief_generation' and st.session_state.get('generated_brief'):
                is_completed = True
            
            if is_current:
                st.markdown(f"**➤ {icon} {label}**")
            elif is_completed:
                st.markdown(f"✅ {icon} {label}")
            else:
                st.markdown(f"⭕ {icon} {label}")
        
        st.divider()
        
        # Current dataset info
        if st.session_state.get('selected_dataset'):
            st.markdown("### Current Dataset")
            dataset = st.session_state.selected_dataset
            st.info(f"**{dataset['name']}**\n`{dataset['code']}`")
            
            if st.button("🔄 Change Dataset"):
                st.session_state.current_step = 'dataset_selection'
                st.session_state.selected_dataset = None
                st.session_state.analysis_config = {}
                st.session_state.generated_brief = None
                st.rerun()
        
        st.divider()
        
        # OpenAI API Key input
        if not st.session_state.get('openai_service'):
            st.markdown("### 🔑 OpenAI API Key")
            api_key = st.text_input(
                "Enter API Key",
                type="password",
                help="Required for AI-powered brief generation"
            )
            
            if api_key and api_key != st.session_state.get('openai_api_key'):
                st.session_state.openai_api_key = api_key
                # Reinitialize services
                st.session_state.services_initialized = False
                st.rerun()
        else:
            st.success("🤖 AI Services Active")

def show_dataset_selection():
    """Show dataset selection page."""
    st.title("📊 Dataset Selection")
    st.markdown("Choose a FAOSTAT dataset to generate an analytical brief")
    
    # Check FAOSTAT service
    if 'faostat_service' not in st.session_state:
        st.error("❌ FAOSTAT service not available")
        return
    
    faostat_service = st.session_state.faostat_service
    
    try:
        with st.spinner("🔄 Loading available datasets..."):
            datasets_df = faostat_service.get_available_datasets()
        
        if datasets_df is None or datasets_df.empty:
            st.warning("⚠️ No datasets found")
            return
        
        st.success(f"✅ Found {len(datasets_df)} datasets")
        
        # Convert to list for easier handling
        datasets_list = datasets_df.to_dict('records')
        
        # Dataset selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search and filter
            search_term = st.text_input("🔍 Search datasets", placeholder="Enter keywords...")
            
            # Filter datasets based on search
            if search_term:
                filtered_datasets = [
                    d for d in datasets_list 
                    if search_term.lower() in d.get('DatasetName', '').lower() or 
                       search_term.lower() in d.get('DatasetCode', '').lower()
                ]
            else:
                filtered_datasets = datasets_list
            
            # Display datasets
            if filtered_datasets:
                st.markdown(f"### Available Datasets ({len(filtered_datasets)})")
                
                for dataset in filtered_datasets:
                    with st.expander(f"📁 {dataset.get('DatasetName', 'Unknown')} ({dataset.get('DatasetCode', 'N/A')})"):
                        
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"**Code:** `{dataset.get('DatasetCode', 'N/A')}`")
                            st.markdown(f"**Description:** {dataset.get('DatasetDescription', 'No description available')}")
                            if dataset.get('DateUpdate'):
                                st.markdown(f"**Last Updated:** {dataset.get('DateUpdate')}")
                        
                        with col_b:
                            if st.button(f"Select", key=f"select_{dataset.get('DatasetCode')}"):
                                st.session_state.selected_dataset = {
                                    'code': dataset.get('DatasetCode'),
                                    'name': dataset.get('DatasetName'),
                                    'description': dataset.get('DatasetDescription', ''),
                                    'last_updated': dataset.get('DateUpdate', '')
                                }
                                st.session_state.current_step = 'configuration'
                                st.rerun()
            else:
                st.info("No datasets match your search criteria")
        
        with col2:
            st.markdown("### Quick Select")
            
            # Popular datasets
            popular_datasets = [
                'QCL',  # Crops and livestock products
                'RFN',  # Fertilizers by Nutrient
                'TCL',  # Crops and livestock products trade
                'EF',   # Environment and Emissions
                'GT'    # Government tier
            ]
            
            st.markdown("**Popular Datasets:**")
            for code in popular_datasets:
                matching_dataset = next((d for d in datasets_list if d.get('DatasetCode') == code), None)
                if matching_dataset:
                    if st.button(f"📊 {code}", key=f"quick_{code}", use_container_width=True):
                        st.session_state.selected_dataset = {
                            'code': matching_dataset.get('DatasetCode'),
                            'name': matching_dataset.get('DatasetName'),
                            'description': matching_dataset.get('DatasetDescription', ''),
                            'last_updated': matching_dataset.get('DateUpdate', '')
                        }
                        st.session_state.current_step = 'configuration'
                        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")

def show_configuration():
    """Show analysis configuration page."""
    st.title("⚙️ Analysis Configuration")
    
    if not st.session_state.get('selected_dataset'):
        st.warning("Please select a dataset first")
        return
    
    dataset = st.session_state.selected_dataset
    st.info(f"Configuring analysis for: **{dataset['name']}** (`{dataset['code']}`)")
    
    # Load dataset to get available options
    faostat_service = st.session_state.faostat_service
    geography_service = st.session_state.geography_service
    
    try:
        with st.spinner("🔄 Loading dataset..."):
            df, metadata = faostat_service.get_dataset(dataset['code'])
        
        if df is None or df.empty:
            st.error("❌ Could not load dataset")
            return
        
        st.success(f"✅ Dataset loaded: {len(df):,} rows")
        
        # Configuration form
        with st.form("analysis_config"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Time Period")
                
                if 'Year' in df.columns:
                    min_year = int(df['Year'].min())
                    max_year = int(df['Year'].max())
                    
                    year_range = st.slider(
                        "Select year range",
                        min_value=min_year,
                        max_value=max_year,
                        value=(max(min_year, max_year - 10), max_year),
                        help=f"Dataset covers {min_year} to {max_year}"
                    )
                else:
                    year_range = None
                    st.info("No year information available")
                
                st.markdown("### 🌾 Items")
                
                if 'Item' in df.columns:
                    available_items = sorted(df['Item'].unique())
                    selected_items = st.multiselect(
                        "Select items to analyze",
                        options=available_items,
                        default=available_items[:3] if len(available_items) > 3 else available_items,
                        help="Choose specific items for analysis"
                    )
                else:
                    selected_items = None
                    st.info("No item information available")
            
            with col2:
                st.markdown("### 🌍 Geographic Scope")
                
                if 'Area' in df.columns:
                    # Geographic classification
                    main_regions = geography_service.get_main_regions()
                    other_regions = geography_service.get_other_regions()
                    countries = geography_service.get_countries()
                    
                    geo_scope = st.radio(
                        "Geographic level",
                        ["Main Regions", "Other Regions", "Countries"]
                    )
                    
                    if geo_scope == "Main Regions":
                        available_areas = [area for area in df['Area'].unique() if area in main_regions]
                        selected_areas = st.multiselect(
                            "Select main regions",
                            options=available_areas,
                            default=available_areas[:3] if len(available_areas) > 3 else available_areas
                        )
                    elif geo_scope == "Other Regions":
                        available_areas = [area for area in df['Area'].unique() if area in other_regions]
                        selected_areas = st.multiselect(
                            "Select regions",
                            options=available_areas,
                            default=available_areas[:5] if len(available_areas) > 5 else available_areas
                        )
                    else:  # Countries
                        available_countries = [area for area in df['Area'].unique() if area in countries]
                        
                        country_selection = st.radio(
                            "Country selection method",
                            ["Top N", "Bottom N", "Specific selection"]
                        )
                        
                        if country_selection in ["Top N", "Bottom N"]:
                            n_countries = st.slider("Number of countries", 5, 20, 10)
                            
                            # Calculate rankings based on total value
                            if 'Value' in df.columns:
                                country_totals = df[df['Area'].isin(available_countries)].groupby('Area')['Value'].sum()
                                if country_selection == "Top N":
                                    selected_areas = country_totals.nlargest(n_countries).index.tolist()
                                else:
                                    selected_areas = country_totals.nsmallest(n_countries).index.tolist()
                                
                                st.info(f"Selected {country_selection.lower()}: {', '.join(selected_areas[:3])}...")
                            else:
                                selected_areas = available_countries[:n_countries]
                        else:
                            selected_areas = st.multiselect(
                                "Select specific countries",
                                options=available_countries,
                                default=available_countries[:5] if len(available_countries) > 5 else available_countries
                            )
                else:
                    selected_areas = None
                    st.info("No area information available")
                
                st.markdown("### ⚡ Elements")
                
                if 'Element' in df.columns:
                    available_elements = sorted(df['Element'].unique())
                    selected_elements = st.multiselect(
                        "Select elements to analyze",
                        options=available_elements,
                        default=available_elements[:2] if len(available_elements) > 2 else available_elements,
                        help="Choose specific elements for analysis"
                    )
                else:
                    selected_elements = None
                    st.info("No element information available")
            
            # Submit configuration
            if st.form_submit_button("🚀 Generate Brief", use_container_width=True):
                config = {
                    'year_range': year_range,
                    'selected_items': selected_items,
                    'selected_areas': selected_areas,
                    'selected_elements': selected_elements,
                    'geo_scope': geo_scope,
                    'dataset_df': df,
                    'dataset_metadata': metadata
                }
                
                st.session_state.analysis_config = config
                st.session_state.current_step = 'brief_generation'
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")

def show_brief_generation():
    """Show brief generation page."""
    st.title("📝 Brief Generation")
    
    if not st.session_state.get('analysis_config'):
        st.warning("Please configure analysis first")
        return
    
    config = st.session_state.analysis_config
    dataset = st.session_state.selected_dataset
    
    st.info(f"Generating analytical brief for: **{dataset['name']}**")
    
    # Check if we have AI services
    if not st.session_state.get('brief_generator'):
        st.error("❌ AI services not available. Please add OpenAI API key.")
        return
    
    brief_generator = st.session_state.brief_generator
    
    if not st.session_state.get('generated_brief'):
        
        with st.spinner("🤖 Generating analytical brief..."):
            try:
                # Generate the brief
                brief_results = brief_generator.generate_brief(
                    dataset_code=dataset['code'],
                    dataset_name=dataset['name'],
                    config=config
                )
                
                st.session_state.generated_brief = brief_results
                
            except Exception as e:
                st.error(f"❌ Error generating brief: {str(e)}")
                return
    
    # Display generated brief
    brief_results = st.session_state.generated_brief
    
    if brief_results:
        st.success("✅ Brief generated successfully!")
        
        # Display the brief content
        st.markdown("## 📄 Generated Analytical Brief")
        
        # Display insights
        if 'insights' in brief_results:
            insights = brief_results['insights']
            
            if 'highlights' in insights:
                st.markdown("### 🔍 Key Highlights")
                st.markdown(insights['highlights'])
            
            if 'global_trends' in insights:
                st.markdown("### 🌍 Global Trends")
                st.markdown(insights['global_trends'])
            
            if 'regional_analysis' in insights:
                st.markdown("### 🗺️ Regional Analysis")
                st.markdown(insights['regional_analysis'])
            
            if 'country_insights' in insights:
                st.markdown("### 🏛️ Country Analysis")
                st.markdown(insights['country_insights'])
        
        # Display figures
        if 'figures' in brief_results and brief_results['figures']:
            st.markdown("### 📊 Figures")
            
            for i, figure in enumerate(brief_results['figures'], 1):
                st.markdown(f"#### Figure {i}: {figure.title}")
                
                if os.path.exists(figure.filepath):
                    st.image(figure.filepath, caption=figure.description)
                else:
                    st.warning(f"Figure {i} not found")
        
        # Navigation to export
        if st.button("📁 Export Brief", type="primary", use_container_width=True):
            st.session_state.current_step = 'export'
            st.rerun()
    
    # Regenerate option
    if st.button("🔄 Regenerate Brief"):
        st.session_state.generated_brief = None
        st.rerun()

def show_export():
    """Show export options page."""
    st.title("📁 Export Options")
    
    if not st.session_state.get('generated_brief'):
        st.warning("Please generate a brief first")
        return
    
    brief_results = st.session_state.generated_brief
    export_service = st.session_state.export_service
    
    st.success("✅ Brief ready for export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Document Export")
        
        doc_format = st.radio(
            "Choose document format",
            ["PDF", "Word"]
        )
        
        if st.button(f"📥 Download {doc_format}", use_container_width=True):
            try:
                with st.spinner(f"🔄 Generating {doc_format}..."):
                    if doc_format == "PDF":
                        file_path = export_service.export_pdf(brief_results)
                    else:
                        file_path = export_service.export_word(brief_results)
                
                # Provide download
                with open(file_path, 'rb') as f:
                    st.download_button(
                        f"📥 Download {doc_format}",
                        data=f.read(),
                        file_name=os.path.basename(file_path),
                        mime="application/pdf" if doc_format == "PDF" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating {doc_format}: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Data Export")
        
        if st.button("📥 Download Excel Data", use_container_width=True):
            try:
                with st.spinner("🔄 Generating Excel file..."):
                    excel_path = export_service.export_excel_data(brief_results)
                
                # Provide download
                with open(excel_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Excel Data",
                        data=f.read(),
                        file_name=os.path.basename(excel_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating Excel: {str(e)}")
    
    # Start new analysis
    st.divider()
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.current_step = 'dataset_selection'
        st.session_state.selected_dataset = None
        st.session_state.analysis_config = {}
        st.session_state.generated_brief = None
        st.rerun()

def main():
    """Main application function."""
    
    # Initialize services
    initialize_services()
    
    # Create sidebar
    create_sidebar()
    
    # Route to appropriate page based on current step
    current_step = st.session_state.get('current_step', 'dataset_selection')
    
    if current_step == 'dataset_selection':
        show_dataset_selection()
    elif current_step == 'configuration':
        show_configuration()
    elif current_step == 'brief_generation':
        show_brief_generation()
    elif current_step == 'export':
        show_export()

if __name__ == "__main__":
    main()