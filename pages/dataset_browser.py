"""
Dataset browser page for FAOSTAT Analytics Application.

This module provides an interface for browsing and selecting FAOSTAT datasets
with streamlined functionality that works with the updated FAOSTAT service.
"""

import streamlit as st
import pandas as pd
import logging
import traceback
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def show_dataset_browser():
    """Main render function for the dataset browser page."""
    
    st.title("📊 Dataset Browser")
    st.markdown("Explore and select FAOSTAT datasets for analysis")
    
    # Check FAOSTAT service availability
    if not check_faostat_service():
        return
    
    # Main dataset browser interface
    show_main_browser_interface()

def check_faostat_service() -> bool:
    """Check if FAOSTAT service is available and working."""
    
    if 'faostat_service' not in st.session_state:
        st.error("❌ **FAOSTAT Service Not Available**")
        st.markdown("""
        The FAOSTAT service is not initialized. This could be due to:
        1. Service initialization failed during app startup
        2. Network connectivity issues
        3. Configuration problems
        """)
        
        if st.button("🔄 Try to Reinitialize FAOSTAT Service"):
            try_reinitialize_faostat_service()
        
        return False
    
    return True

def try_reinitialize_faostat_service():
    """Try to reinitialize the FAOSTAT service."""
    
    try:
        from services.faostat_service import FAOStatService
        from config.config_manager import ConfigManager
        
        st.info("🔄 Attempting to reinitialize FAOSTAT service...")
        
        # Get configuration
        if 'config' in st.session_state:
            config = st.session_state.config
        else:
            config = ConfigManager()
            st.session_state.config = config
        
        # Create new FAOSTAT service
        faostat_service = FAOStatService(
            base_url=config.get_faostat_base_url(),
            cache_duration=config.get_int('faostat.cache_duration', 3600)
        )
        
        # Test the service
        st.info("🧪 Testing service...")
        datasets_df = faostat_service.get_available_datasets()
        
        if datasets_df is not None and not datasets_df.empty:
            st.session_state.faostat_service = faostat_service
            st.success(f"✅ FAOSTAT service reinitialized successfully! Found {len(datasets_df)} datasets")
            st.rerun()
        else:
            st.error("❌ Service initialized but no datasets found")
            
    except Exception as e:
        st.error(f"❌ Failed to reinitialize service: {str(e)}")
        st.code(traceback.format_exc())

def show_main_browser_interface():
    """Show the main dataset browser interface."""
    
    st.subheader("📊 Available Datasets")
    
    # Get FAOSTAT service
    faostat_service = st.session_state.faostat_service
    
    # Add debug toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")
    
    # Load datasets
    try:
        with st.spinner("🔄 Loading available datasets..."):
            datasets_df = load_datasets(faostat_service, debug_mode)
        
        if datasets_df is None or datasets_df.empty:
            st.warning("⚠️ No datasets found")
            show_dataset_troubleshooting()
            return
        
        st.success(f"✅ Found {len(datasets_df)} datasets")
        
        # Convert DataFrame to list of dictionaries for easier handling
        datasets_list = datasets_df.to_dict('records')
        
        # Dataset filtering and search
        filtered_datasets = show_dataset_filters(datasets_list)
        
        # Dataset list/grid
        show_dataset_list(filtered_datasets)
        
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")
        if debug_mode:
            st.code(traceback.format_exc())
        
        if st.button("🔄 Retry Loading Datasets"):
            st.rerun()

def load_datasets(faostat_service, debug_mode: bool = False) -> Optional[pd.DataFrame]:
    """Load datasets using the updated FAOSTAT service."""
    
    try:
        if debug_mode:
            st.write("**Debug: Loading datasets...**")
            st.write(f"Service type: {type(faostat_service)}")
        
        # Call the service method (returns DataFrame)
        datasets_df = faostat_service.get_available_datasets(force_refresh=False)
        
        if debug_mode:
            st.write(f"Result type: {type(datasets_df)}")
            if datasets_df is not None:
                st.write(f"DataFrame shape: {datasets_df.shape}")
                st.write(f"DataFrame columns: {list(datasets_df.columns)}")
                if not datasets_df.empty:
                    st.write("**Sample dataset:**")
                    st.write(datasets_df.iloc[0].to_dict())
        
        return datasets_df
        
    except Exception as e:
        st.error(f"Exception in load_datasets: {str(e)}")
        if debug_mode:
            st.code(traceback.format_exc())
        return None

def show_dataset_filters(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Show dataset filtering options and return filtered datasets."""
    
    st.subheader("🔍 Filter Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Search by name/description
        search_term = st.text_input(
            "🔍 Search datasets",
            placeholder="Enter dataset name or description...",
            help="Search in dataset names and descriptions"
        )
    
    with col2:
        # Category filter - extract categories from descriptions or codes
        categories = set()
        for dataset in datasets:
            # Try to infer category from code or description
            code = dataset.get('code', '')
            if code:
                if code.startswith('Q'):
                    categories.add('Production')
                elif code.startswith('T'):
                    categories.add('Trade')
                elif code.startswith('F'):
                    categories.add('Food Security')
                elif code.startswith('E'):
                    categories.add('Environment')
                elif code.startswith('R'):
                    categories.add('Resources')
                else:
                    categories.add('Other')
            else:
                categories.add('Other')
        
        selected_category = st.selectbox(
            "📂 Category",
            options=["All"] + sorted(list(categories)),
            help="Filter by dataset category"
        )
    
    # Apply filters
    filtered_datasets = datasets
    
    if search_term:
        search_lower = search_term.lower()
        filtered_datasets = [
            d for d in filtered_datasets 
            if search_lower in d.get('name', '').lower() 
            or search_lower in d.get('description', '').lower()
            or search_lower in d.get('code', '').lower()
        ]
    
    if selected_category != "All":
        def get_category(dataset):
            code = dataset.get('code', '')
            if code.startswith('Q'):
                return 'Production'
            elif code.startswith('T'):
                return 'Trade'
            elif code.startswith('F'):
                return 'Food Security'
            elif code.startswith('E'):
                return 'Environment'
            elif code.startswith('R'):
                return 'Resources'
            else:
                return 'Other'
        
        filtered_datasets = [
            d for d in filtered_datasets 
            if get_category(d) == selected_category
        ]
    
    st.info(f"Showing {len(filtered_datasets)} of {len(datasets)} datasets")
    
    return filtered_datasets

def show_dataset_list(datasets: List[Dict[str, Any]]):
    """Show the filtered dataset list."""
    
    st.subheader("📋 Dataset List")
    
    if not datasets:
        st.info("No datasets match your filters")
        return
    
    # Pagination
    items_per_page = 10
    total_pages = (len(datasets) - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.slider("Page", 1, total_pages, 1) - 1
        start_idx = page * items_per_page
        end_idx = start_idx + items_per_page
        page_datasets = datasets[start_idx:end_idx]
    else:
        page_datasets = datasets
    
    # Display datasets
    for i, dataset in enumerate(page_datasets):
        with st.expander(f"📊 {dataset.get('code', 'N/A')} - {dataset.get('name', 'Unknown Dataset')}"):
            show_dataset_details(dataset)

def show_dataset_details(dataset: Dict[str, Any]):
    """Show detailed information about a dataset."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Description:** {dataset.get('description', 'No description available')}")
        st.markdown(f"**Code:** `{dataset.get('code', 'N/A')}`")
        st.markdown(f"**Last Updated:** {dataset.get('last_updated', 'Unknown')}")
        
        # Show category
        code = dataset.get('code', '')
        category = get_dataset_category(code)
        st.markdown(f"**Category:** {category}")
    
    with col2:
        st.markdown("**Actions:**")
        
        dataset_code = dataset.get('code')
        
        col1_action, col2_action = st.columns(2)
        
        with col1_action:
            if st.button(f"👁️ Preview", key=f"preview_{dataset_code}", use_container_width=True):
                preview_dataset(dataset)
        
        with col2_action:
            if st.button(f"📥 Load", key=f"load_{dataset_code}", use_container_width=True):
                load_selected_dataset(dataset)

def get_dataset_category(code: str) -> str:
    """Get dataset category based on code."""
    if not code:
        return 'Other'
    
    if code.startswith('Q'):
        return 'Production & Quantities'
    elif code.startswith('T'):
        return 'Trade'
    elif code.startswith('F'):
        return 'Food Security'
    elif code.startswith('E'):
        return 'Environment'
    elif code.startswith('R'):
        return 'Resources'
    elif code.startswith('P'):
        return 'Prices'
    elif code.startswith('I'):
        return 'Indicators'
    else:
        return 'Other'

def preview_dataset(dataset: Dict[str, Any]):
    """Preview a dataset without fully loading it."""
    
    dataset_code = dataset.get('code')
    st.info(f"🔍 Previewing dataset: {dataset_code}")
    
    try:
        faostat_service = st.session_state.faostat_service
        
        # Get a preview
        with st.spinner("Loading preview..."):
            df, metadata = faostat_service.get_dataset_preview(dataset_code, max_rows=50)
        
        if df is not None and not df.empty:
            st.success(f"✅ Preview loaded: {len(df)} sample rows")
            
            # FIXED: Show basic info without nested columns - use metrics in a single row
            st.markdown("### 📊 Dataset Overview")
            
            # Create metrics safely without nested columns
            metrics_data = {
                "Sample Rows": len(df),
                "Columns": len(df.columns),
                "Year Range": f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A"
            }
            
            # Display metrics as text instead of columns to avoid nesting
            for label, value in metrics_data.items():
                st.text(f"{label}: {value}")
            
            # Show column info
            st.markdown("### 📋 Dataset Structure")
            st.write("**Columns:**", ", ".join(df.columns.tolist()))
            
            # Show sample data
            st.markdown("### 📄 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            st.error("❌ Could not load preview data")
            
    except Exception as e:
        st.error(f"❌ Preview failed: {str(e)}")
        logger.error(f"Preview error for {dataset_code}: {str(e)}")

def load_selected_dataset(dataset: Dict[str, Any]):
    """Load the selected dataset into session state."""
    
    dataset_code = dataset.get('code')
    dataset_name = dataset.get('name', 'Unknown Dataset')
    
    st.info(f"📥 Loading dataset: {dataset_name}")
    
    try:
        faostat_service = st.session_state.faostat_service
        
        with st.spinner("Loading full dataset..."):
            df, metadata = faostat_service.get_dataset(dataset_code)
        
        if df is not None and not df.empty:
            # Store in session state
            st.session_state.current_dataset = dataset_code
            st.session_state.current_dataset_df = df
            st.session_state.current_dataset_metadata = metadata
            st.session_state.current_dataset_name = dataset_name
            
            st.success(f"✅ Dataset loaded successfully!")
            st.success(f"📊 {len(df):,} rows × {len(df.columns)} columns")
            
            # FIXED: Show basic stats without nested columns
            st.markdown("### 📊 Dataset Summary")
            
            # Create summary safely without columns
            summary_info = []
            if 'Year' in df.columns:
                summary_info.append(f"**Year Range:** {df['Year'].min()} - {df['Year'].max()}")
            if 'Area' in df.columns:
                summary_info.append(f"**Countries:** {df['Area'].nunique()}")
            if 'Item' in df.columns:
                summary_info.append(f"**Items:** {df['Item'].nunique()}")
            
            for info in summary_info:
                st.markdown(info)
            
            # Show what to do next
            st.info("💡 **Next Steps:** Use the sidebar navigation to go to Analysis or Natural Language Queries to start exploring your data!")
            
        else:
            st.error("❌ Failed to load dataset - no data returned")
            
    except Exception as e:
        st.error(f"❌ Loading failed: {str(e)}")
        logger.error(f"Load error for {dataset_code}: {str(e)}")

def show_dataset_troubleshooting():
    """Show troubleshooting options when no datasets are found."""
    
    st.markdown("### 🔧 Troubleshooting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Force Refresh", use_container_width=True):
            # Clear any cached data and retry
            if hasattr(st.session_state.faostat_service, '_cache'):
                st.session_state.faostat_service._cache.clear()
            st.rerun()
    
    with col2:
        if st.button("🔧 Reinitialize Service", use_container_width=True):
            try_reinitialize_faostat_service()
    
    st.markdown("""
    **Common Issues:**
    1. **Network Problems**: FAOSTAT servers might be temporarily unavailable
    2. **API Changes**: The FAOSTAT API structure might have changed
    3. **Cache Issues**: Cached data might be corrupted
    
    **Manual Test:**
    Try visiting the FAOSTAT datasets URL directly: 
    [https://bulks-faostat.fao.org/production/datasets_E.json](https://bulks-faostat.fao.org/production/datasets_E.json)
    """)

if __name__ == "__main__":
    show_dataset_browser()
    