"""
Dataset browser page for the FAOSTAT Analytics application.

This module provides an interface for exploring available FAOSTAT datasets,
previewing their contents, and selecting datasets for analysis.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def render():
    """Render the dataset browser page."""
    
    st.title("📊 Dataset Browser")
    st.markdown("Explore and select from 200+ FAOSTAT datasets for analysis")
    
    # Check if FAOSTAT service is available
    if not hasattr(st.session_state, 'faostat_service'):
        st.error("❌ FAOSTAT service not available. Please restart the application.")
        return
    
    faostat_service = st.session_state.faostat_service
    
    # Load datasets
    with st.spinner("Loading available datasets..."):
        datasets_df = load_datasets(faostat_service)
    
    if datasets_df.empty:
        st.error("❌ No datasets available. Please check your connection.")
        return
    
    # Dataset filtering and search
    filtered_df = render_dataset_filters(datasets_df)
    
    # Dataset list
    render_dataset_list(filtered_df, faostat_service)
    
    # Dataset preview
    if st.session_state.get('selected_dataset_code'):
        render_dataset_preview(faostat_service)

def load_datasets(faostat_service) -> pd.DataFrame:
    """Load available datasets from FAOSTAT."""
    try:
        return faostat_service.get_available_datasets()
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        st.error(f"Error loading datasets: {str(e)}")
        return pd.DataFrame()

def render_dataset_filters(datasets_df: pd.DataFrame) -> pd.DataFrame:
    """Render dataset filtering interface."""
    
    st.markdown("## 🔍 Find Datasets")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search by name or description
        search_term = st.text_input(
            "🔎 Search datasets",
            placeholder="e.g., production, trade, population, fertilizer...",
            help="Search in dataset names and descriptions"
        )
    
    with col2:
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            options=["name", "last_updated", "code"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    
    # Filter datasets based on search
    filtered_df = datasets_df.copy()
    
    if search_term:
        mask = (
            filtered_df['name'].str.contains(search_term, case=False, na=False) |
            filtered_df['description'].str.contains(search_term, case=False, na=False) |
            filtered_df['code'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Sort datasets
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by)
    
    # Show filter results
    st.info(f"📊 Showing {len(filtered_df)} of {len(datasets_df)} datasets")
    
    return filtered_df

def render_dataset_list(datasets_df: pd.DataFrame, faostat_service) -> None:
    """Render the list of available datasets."""
    
    st.markdown("## 📋 Available Datasets")
    
    if datasets_df.empty:
        st.warning("No datasets match your search criteria.")
        return
    
    # Pagination
    datasets_per_page = 10
    total_pages = (len(datasets_df) - 1) // datasets_per_page + 1
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
    else:
        page = 1
    
    # Calculate page range
    start_idx = (page - 1) * datasets_per_page
    end_idx = min(start_idx + datasets_per_page, len(datasets_df))
    page_datasets = datasets_df.iloc[start_idx:end_idx]
    
    # Display datasets
    for idx, row in page_datasets.iterrows():
        render_dataset_card(row, faostat_service)

def render_dataset_card(dataset_row: pd.Series, faostat_service) -> None:
    """Render a single dataset card."""
    
    with st.container():
        # Create a bordered container
        st.markdown("""
        <style>
        .dataset-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### 📊 {dataset_row['name']}")
            st.markdown(f"**Code:** `{dataset_row['code']}`")
            st.markdown(f"**Description:** {dataset_row['description']}")
            if dataset_row['last_updated']:
                st.markdown(f"**Last Updated:** {dataset_row['last_updated']}")
        
        with col2:
            # Preview button
            if st.button(f"👀 Preview", key=f"preview_{dataset_row['code']}"):
                st.session_state.selected_dataset_code = dataset_row['code']
                st.session_state.selected_dataset_name = dataset_row['name']
                st.rerun()
        
        with col3:
            # Select for analysis button
            if st.button(f"📈 Analyze", key=f"analyze_{dataset_row['code']}"):
                st.session_state.current_dataset_code = dataset_row['code']
                st.session_state.current_dataset_name = dataset_row['name']
                st.session_state.selected_page = "Analysis"
                st.success(f"✅ Selected: {dataset_row['name']}")
                st.rerun()
        
        # Show if this is the currently selected dataset
        if st.session_state.get('current_dataset_code') == dataset_row['code']:
            st.success("🎯 Currently selected for analysis")
        
        st.markdown("---")

def render_dataset_preview(faostat_service) -> None:
    """Render dataset preview section."""
    
    dataset_code = st.session_state.selected_dataset_code
    dataset_name = st.session_state.get('selected_dataset_name', dataset_code)
    
    st.markdown(f"## 👀 Dataset Preview: {dataset_name}")
    st.markdown(f"**Code:** `{dataset_code}`")
    
    # Load preview data
    with st.spinner(f"Loading preview for {dataset_code}..."):
        try:
            preview_df, metadata = faostat_service.get_dataset_preview(dataset_code, max_rows=50)
            
            if preview_df is not None and not preview_df.empty:
                render_preview_content(preview_df, metadata, faostat_service)
            else:
                st.error("❌ Could not load dataset preview")
                
        except Exception as e:
            logger.error(f"Error loading preview for {dataset_code}: {str(e)}")
            st.error(f"Error loading preview: {str(e)}")

def render_preview_content(preview_df: pd.DataFrame, metadata: Dict, faostat_service) -> None:
    """Render the preview content for a dataset."""
    
    # Dataset summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows (Preview)", len(preview_df))
    
    with col2:
        st.metric("Columns", len(preview_df.columns))
    
    with col3:
        if 'Area' in preview_df.columns:
            st.metric("Countries/Areas", preview_df['Area'].nunique())
        else:
            st.metric("Countries/Areas", "N/A")
    
    with col4:
        if 'Year' in preview_df.columns:
            year_range = f"{preview_df['Year'].min()}-{preview_df['Year'].max()}"
            st.metric("Year Range", year_range)
        else:
            st.metric("Year Range", "N/A")
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Column Analysis", "ℹ️ Metadata"])
    
    with tab1:
        render_data_preview_tab(preview_df)
    
    with tab2:
        render_column_analysis_tab(preview_df)
    
    with tab3:
        render_metadata_tab(metadata, preview_df)

def render_data_preview_tab(preview_df: pd.DataFrame) -> None:
    """Render the data preview tab."""
    
    st.markdown("### Sample Data")
    st.dataframe(preview_df, use_container_width=True)
    
    # Download option
    if st.button("💾 Export Preview as CSV"):
        csv = preview_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"faostat_preview_{st.session_state.selected_dataset_code}.csv",
            mime="text/csv"
        )

def render_column_analysis_tab(preview_df: pd.DataFrame) -> None:
    """Render the column analysis tab."""
    
    st.markdown("### Column Information")
    
    # Column summary
    column_info = []
    for col in preview_df.columns:
        col_data = {
            'Column': col,
            'Type': str(preview_df[col].dtype),
            'Non-null Count': preview_df[col].count(),
            'Unique Values': preview_df[col].nunique(),
            'Sample Values': ', '.join(str(x) for x in preview_df[col].dropna().unique()[:3])
        }
        column_info.append(col_data)
    
    column_df = pd.DataFrame(column_info)
    st.dataframe(column_df, use_container_width=True)
    
    # Show unique values for categorical columns
    categorical_columns = preview_df.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) > 0:
        st.markdown("### Categorical Columns Details")
        
        selected_col = st.selectbox(
            "Select column to explore",
            options=categorical_columns
        )
        
        if selected_col:
            unique_values = preview_df[selected_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Top values in {selected_col}:**")
                st.dataframe(unique_values.head(10))
            
            with col2:
                # Simple bar chart of top values
                if len(unique_values) > 0:
                    st.bar_chart(unique_values.head(10))

def render_metadata_tab(metadata: Dict, preview_df: pd.DataFrame) -> None:
    """Render the metadata tab."""
    
    st.markdown("### Dataset Metadata")
    
    if metadata:
        # Display metadata in a structured way
        for key, value in metadata.items():
            if key != 'fileSize':  # Skip complex nested structures
                st.text(f"{key}: {value}")
    else:
        st.info("No metadata available for this dataset")
    
    # Generate summary statistics
    st.markdown("### Summary Statistics")
    
    if not preview_df.empty:
        # Generate comprehensive summary
        summary_stats = generate_summary_statistics(preview_df)
        
        for section, stats in summary_stats.items():
            st.markdown(f"**{section}:**")
            for stat_name, stat_value in stats.items():
                st.text(f"  {stat_name}: {stat_value}")

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Generate comprehensive summary statistics for the dataset."""
    
    stats = {}
    
    # Basic information
    stats['Basic Information'] = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
    }
    
    # Column type breakdown
    type_counts = df.dtypes.value_counts()
    stats['Column Types'] = {str(dtype): count for dtype, count in type_counts.items()}
    
    # Missing data information
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        stats['Missing Data'] = {col: f"{count} ({count/len(df)*100:.1f}%)" 
                               for col, count in missing_data.items()}
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats['Numeric Columns'] = {
            'Count': len(numeric_cols),
            'Columns': ', '.join(numeric_cols)
        }
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        stats['Categorical Columns'] = {
            'Count': len(categorical_cols),
            'Columns': ', '.join(categorical_cols)
        }
    
    # Date/time information
    if 'Year' in df.columns:
        stats['Time Coverage'] = {
            'Earliest Year': str(df['Year'].min()),
            'Latest Year': str(df['Year'].max()),
            'Year Count': str(df['Year'].nunique())
        }
    
    # Geographic coverage
    if 'Area' in df.columns:
        stats['Geographic Coverage'] = {
            'Countries/Areas': str(df['Area'].nunique()),
            'Top 3 Areas': ', '.join(df['Area'].value_counts().head(3).index)
        }
    
    # Item coverage
    if 'Item' in df.columns:
        stats['Item Coverage'] = {
            'Total Items': str(df['Item'].nunique()),
            'Top 3 Items': ', '.join(df['Item'].value_counts().head(3).index)
        }
    
    return stats