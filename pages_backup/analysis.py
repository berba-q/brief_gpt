"""
Analysis page for the FAOSTAT Analytics application.

This module provides the main analysis interface where users can filter data,
generate visualizations, and create AI-powered insights.
"""

import os
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from models.data_models import AnalysisResults, AnalysisFilter
from utils.data_processing import identify_top_bottom_performers

logger = logging.getLogger(__name__)

def show_analysis_page():
    """Render the analysis page."""
    
    st.title("🔬 Data Analysis")
    
    # Check if we have necessary services
    if not hasattr(st.session_state, 'faostat_service'):
        st.error("❌ FAOSTAT service not available. Please restart the application.")
        return
    
    # Check if a dataset is selected
    current_dataset_code = st.session_state.get('current_dataset_code')
    
    if not current_dataset_code:
        render_dataset_selection()
        return
    
    # Load and analyze dataset
    render_analysis_interface(current_dataset_code)

def render_dataset_selection():
    """Render dataset selection interface."""
    
    st.markdown("## 📊 Select Dataset for Analysis")
    st.info("👆 Please select a dataset from the Dataset Browser first, or choose a quick option below.")
    
    # Quick dataset selection
    faostat_service = st.session_state.faostat_service
    
    with st.spinner("Loading popular datasets..."):
        try:
            datasets_df = faostat_service.get_available_datasets()
            
            if not datasets_df.empty:
                # Show some popular datasets for quick selection
                popular_datasets = [
                    "QCL", "FBS", "RFN", "TCL", "OA"  # Common dataset codes
                ]
                
                available_popular = datasets_df[
                    datasets_df['code'].isin(popular_datasets)
                ]
                
                if not available_popular.empty:
                    st.markdown("### 🚀 Quick Start - Popular Datasets")
                    
                    for _, row in available_popular.head(5).iterrows():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{row['name']}** (`{row['code']}`)")
                            st.markdown(f"_{row['description'][:100]}..._")
                        
                        with col2:
                            if st.button(f"Select", key=f"quick_{row['code']}"):
                                st.session_state.current_dataset_code = row['code']
                                st.session_state.current_dataset_name = row['name']
                                st.rerun()
        
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            st.error("Could not load datasets. Please try again.")
    
    # Link to dataset browser
    st.markdown("### 🔍 Or browse all datasets")
    if st.button("📊 Go to Dataset Browser"):
        st.session_state.selected_page = "Dataset Browser"
        st.rerun()

def render_analysis_interface(dataset_code: str):
    """Render the main analysis interface."""
    
    dataset_name = st.session_state.get('current_dataset_name', dataset_code)
    
    st.markdown(f"## 📊 Analyzing: {dataset_name}")
    st.markdown(f"**Dataset Code:** `{dataset_code}`")
    
    # Load dataset
    dataset_df = load_dataset(dataset_code)
    
    if dataset_df is None or dataset_df.empty:
        st.error("❌ Could not load dataset or dataset is empty.")
        return
    
    # Store dataset in session state
    st.session_state.current_dataset = dataset_df
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Configure Analysis", 
        "📊 View Data", 
        "📈 Visualizations", 
        "🤖 AI Insights"
    ])
    
    with tab1:
        filters = render_analysis_configuration(dataset_df)
    
    with tab2:
        render_data_view(dataset_df, filters)
    
    with tab3:
        render_visualizations(dataset_df, filters, dataset_name)
    
    with tab4:
        render_ai_insights(dataset_df, filters, dataset_name)

def load_dataset(dataset_code: str) -> Optional[pd.DataFrame]:
    """Load the full dataset."""
    
    faostat_service = st.session_state.faostat_service
    
    with st.spinner(f"Loading dataset {dataset_code}..."):
        try:
            df, metadata = faostat_service.get_dataset(dataset_code)
            
            if df is not None:
                st.success(f"✅ Loaded {len(df):,} records")
                
                # Store metadata
                st.session_state.current_dataset_metadata = metadata
                
                return df
            else:
                st.error("Failed to load dataset")
                return None
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_code}: {str(e)}")
            st.error(f"Error loading dataset: {str(e)}")
            return None

def render_analysis_configuration(dataset_df: pd.DataFrame) -> AnalysisFilter:
    """Render analysis configuration interface."""
    
    st.markdown("### ⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Time Period")
        
        # Year range selection
        if 'Year' in dataset_df.columns:
            min_year = int(dataset_df['Year'].min())
            max_year = int(dataset_df['Year'].max())
            
            year_range = st.slider(
                "Select year range",
                min_value=min_year,
                max_value=max_year,
                value=(max(min_year, max_year - 10), max_year),
                help=f"Dataset covers {min_year} to {max_year}"
            )
        else:
            year_range = None
            st.info("No year information available in this dataset")
        
        st.markdown("#### 🌍 Geographic Scope")
        
        # Region selection
        regions = None
        if 'Area' in dataset_df.columns:
            available_areas = sorted(dataset_df['Area'].unique())
            
            region_option = st.radio(
                "Region selection",
                ["All regions", "Select specific", "Top countries only"]
            )
            
            if region_option == "Select specific":
                regions = st.multiselect(
                    "Choose regions/countries",
                    options=available_areas,
                    default=available_areas[:10] if len(available_areas) > 10 else available_areas
                )
            elif region_option == "Top countries only":
                top_n = st.slider("Number of top countries", 5, 20, 10)
                # Get top countries by total value
                if 'Value' in dataset_df.columns:
                    top_countries = dataset_df.groupby('Area')['Value'].sum().nlargest(top_n).index.tolist()
                    regions = top_countries
                    st.info(f"Selected top {len(regions)} countries by total value")
    
    with col2:
        st.markdown("#### 🌾 Items & Elements")
        
        # Item selection
        items = None
        if 'Item' in dataset_df.columns:
            available_items = sorted(dataset_df['Item'].unique())
            
            item_option = st.radio(
                "Item selection",
                ["All items", "Select specific", "Top items only"]
            )
            
            if item_option == "Select specific":
                items = st.multiselect(
                    "Choose items",
                    options=available_items,
                    default=available_items[:5] if len(available_items) > 5 else available_items
                )
            elif item_option == "Top items only":
                top_n_items = st.slider("Number of top items", 3, 15, 5)
                if 'Value' in dataset_df.columns:
                    top_items = dataset_df.groupby('Item')['Value'].sum().nlargest(top_n_items).index.tolist()
                    items = top_items
                    st.info(f"Selected top {len(items)} items by total value")
        
        # Element selection
        elements = None
        if 'Element' in dataset_df.columns:
            available_elements = sorted(dataset_df['Element'].unique())
            
            if len(available_elements) > 1:
                elements = st.multiselect(
                    "Choose elements/indicators",
                    options=available_elements,
                    default=available_elements[:3] if len(available_elements) > 3 else available_elements
                )
            else:
                elements = available_elements
                st.info(f"Single element available: {available_elements[0]}")
    
    # Create filter object
    filters = AnalysisFilter(
        year_range=year_range,
        regions=regions,
        items=items,
        elements=elements
    )
    
    # Show filter summary
    render_filter_summary(filters, dataset_df)
    
    return filters

def render_filter_summary(filters: AnalysisFilter, dataset_df: pd.DataFrame):
    """Render a summary of applied filters."""
    
    st.markdown("### 📋 Filter Summary")
    
    # Apply filters to get filtered dataset size
    filtered_df = apply_filters(dataset_df, filters)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original Records", 
            f"{len(dataset_df):,}",
            help="Total records in the dataset"
        )
    
    with col2:
        st.metric(
            "Filtered Records", 
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(dataset_df):,}"
        )
    
    with col3:
        reduction_pct = (1 - len(filtered_df) / len(dataset_df)) * 100
        st.metric(
            "Data Reduction", 
            f"{reduction_pct:.1f}%",
            help="Percentage of data filtered out"
        )
    
    # Show active filters
    active_filters = []
    if filters.year_range:
        active_filters.append(f"📅 Years: {filters.year_range[0]}-{filters.year_range[1]}")
    if filters.regions:
        active_filters.append(f"🌍 Regions: {len(filters.regions)} selected")
    if filters.items:
        active_filters.append(f"🌾 Items: {len(filters.items)} selected")
    if filters.elements:
        active_filters.append(f"📊 Elements: {len(filters.elements)} selected")
    
    if active_filters:
        st.info("**Active Filters:** " + " | ".join(active_filters))
    else:
        st.info("**No filters applied** - showing all data")

def apply_filters(dataset_df: pd.DataFrame, filters: AnalysisFilter) -> pd.DataFrame:
    """Apply filters to the dataset."""
    
    filtered_df = dataset_df.copy()
    
    # Apply year filter
    if filters.year_range and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Year'] >= filters.year_range[0]) & 
            (filtered_df['Year'] <= filters.year_range[1])
        ]
    
    # Apply region filter
    if filters.regions and 'Area' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Area'].isin(filters.regions)]
    
    # Apply item filter
    if filters.items and 'Item' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Item'].isin(filters.items)]
    
    # Apply element filter
    if filters.elements and 'Element' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Element'].isin(filters.elements)]
    
    return filtered_df

def render_data_view(dataset_df: pd.DataFrame, filters: AnalysisFilter):
    """Render data view tab."""
    
    st.markdown("### 📊 Filtered Dataset")
    
    # Apply filters
    filtered_df = apply_filters(dataset_df, filters)
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
        return
    
    # Data display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_rows = st.selectbox(
            "Rows to display",
            [50, 100, 500, 1000],
            index=1
        )
    
    with col2:
        if st.button("📊 Generate Summary Statistics"):
            render_summary_statistics(filtered_df)
    
    with col3:
        # Download option
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"faostat_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Display data
    st.dataframe(filtered_df.head(show_rows), use_container_width=True)
    
    if len(filtered_df) > show_rows:
        st.info(f"Showing first {show_rows} rows of {len(filtered_df):,} total rows")

def render_summary_statistics(filtered_df: pd.DataFrame):
    """Render summary statistics for the filtered dataset."""
    
    st.markdown("#### 📈 Summary Statistics")
    
    # Generate statistics using the FAOSTAT service
    faostat_service = st.session_state.faostat_service
    summary_stats = faostat_service.generate_summary_stats(filtered_df)
    
    if summary_stats:
        # Display in organized sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview:**")
            st.text(f"• Total Records: {summary_stats.get('row_count', 'N/A'):,}")
            st.text(f"• Columns: {summary_stats.get('column_count', 'N/A')}")
            
            if 'year_range' in summary_stats:
                year_range = summary_stats['year_range']
                st.text(f"• Year Range: {year_range[0]} - {year_range[1]}")
            
            if 'area_count' in summary_stats:
                st.text(f"• Countries/Areas: {summary_stats['area_count']}")
        
        with col2:
            if 'value_stats' in summary_stats:
                value_stats = summary_stats['value_stats']
                st.markdown("**Value Statistics:**")
                st.text(f"• Mean: {value_stats['mean']:,.1f}")
                st.text(f"• Median: {value_stats['median']:,.1f}")
                st.text(f"• Min: {value_stats['min']:,.1f}")
                st.text(f"• Max: {value_stats['max']:,.1f}")
        
        # Time series information
        if 'time_series' in summary_stats:
            ts_info = summary_stats['time_series']
            st.markdown("**Trend Analysis:**")
            st.text(f"• Overall Direction: {ts_info['direction'].title()}")
            st.text(f"• Total Change: {ts_info['percent_change']:.1f}%")
            if 'cagr' in ts_info:
                st.text(f"• CAGR: {ts_info['cagr']:.2f}%")

def render_visualizations(dataset_df: pd.DataFrame, filters: AnalysisFilter, dataset_name: str):
    """Render visualizations tab."""
    
    st.markdown("### 📈 Data Visualizations")
    
    # Apply filters
    filtered_df = apply_filters(dataset_df, filters)
    
    if filtered_df.empty:
        st.warning("⚠️ No data to visualize with current filters.")
        return
    
    # Generate visualizations
    if st.button("🎨 Generate Visualizations", type="primary"):
        with st.spinner("Creating visualizations..."):
            generate_and_display_visualizations(filtered_df, dataset_name)
    
    # Display existing visualizations if any
    display_existing_visualizations()

def generate_and_display_visualizations(filtered_df: pd.DataFrame, dataset_name: str):
    """Generate and display visualizations."""
    
    try:
        viz_generator = st.session_state.viz_generator
        
        # Generate visualizations
        figures = viz_generator.generate_visualizations(filtered_df, dataset_name)
        
        if figures:
            st.success(f"✅ Generated {len(figures)} visualizations")
            
            # Store figures in session state
            st.session_state.current_figures = figures
            
            # Display figures
            for i, figure in enumerate(figures):
                st.markdown(f"#### {figure.title}")
                
                if figure.filepath and os.path.exists(figure.filepath):
                    st.image(figure.filepath, caption=figure.description)
                else:
                    st.error(f"Could not display figure: {figure.filename}")
        else:
            st.warning("No visualizations could be generated with the current data.")
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        st.error(f"Error generating visualizations: {str(e)}")

def display_existing_visualizations():
    """Display previously generated visualizations."""
    
    figures = st.session_state.get('current_figures', [])
    
    if figures:
        st.markdown("#### 🖼️ Generated Visualizations")
        
        for i, figure in enumerate(figures):
            with st.expander(f"📊 {figure.title}", expanded=True):
                
                if figure.filepath and os.path.exists(figure.filepath):
                    st.image(figure.filepath)
                    st.markdown(f"**Description:** {figure.description}")
                    
                    # Download button for individual figure
                    with open(figure.filepath, "rb") as file:
                        st.download_button(
                            label=f"💾 Download {figure.filename}",
                            data=file.read(),
                            file_name=figure.filename,
                            mime="image/png"
                        )
                else:
                    st.error(f"Figure file not found: {figure.filename}")

def render_ai_insights(dataset_df: pd.DataFrame, filters: AnalysisFilter, dataset_name: str):
    """Render AI insights tab."""
    
    st.markdown("### 🤖 AI-Powered Insights")
    
    # Check if OpenAI service is available
    openai_service = st.session_state.get('openai_service')
    
    if not openai_service:
        st.warning("⚠️ OpenAI API key not configured. Please add your API key in the sidebar to enable AI features.")
        return
    
    # Apply filters
    filtered_df = apply_filters(dataset_df, filters)
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for AI analysis with current filters.")
        return
    
    # Generate insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🧠 Generate AI Insights", type="primary"):
            generate_ai_insights(filtered_df, dataset_name, openai_service)
    
    with col2:
        model_type = st.selectbox(
            "AI Model",
            ["comprehensive", "standard"],
            help="Comprehensive: GPT-4 (better quality)\nStandard: GPT-3.5 (faster)"
        )
    
    # Display existing insights
    display_existing_insights()

def generate_ai_insights(filtered_df: pd.DataFrame, dataset_name: str, openai_service):
    """Generate AI insights for the filtered dataset."""
    
    with st.spinner("🤖 AI is analyzing your data..."):
        try:
            # Generate summary statistics
            faostat_service = st.session_state.faostat_service
            summary_stats = faostat_service.generate_summary_stats(filtered_df)
            
            # Get figure descriptions if available
            figures = st.session_state.get('current_figures', [])
            figure_descriptions = [fig.description for fig in figures]
            
            insights = openai_service.generate_insights(
                dataset_name=dataset_name,
                summary_stats=summary_stats,
                figures_descriptions=figure_descriptions
            )
            # --- Patch: parse string JSON if needed ---
            import json
            if isinstance(insights, str):
                try:
                    insights = json.loads(insights)
                except Exception as parse_err:
                    logger.error(f"Failed to parse insights JSON in caller: {parse_err}")
                    st.error("⚠️ AI returned an invalid response. Please try again.")
                    return
            # ------------------------------------------
            # Store insights
            st.session_state.current_insights = insights
            
            st.success("✅ AI analysis complete!")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            st.error(f"Error generating AI insights: {str(e)}")

def display_existing_insights():
    """Display previously generated AI insights."""
    
    insights = st.session_state.get('current_insights')

    import json
    if isinstance(insights, str):
        try:
            insights = json.loads(insights)
        except Exception as e:
            st.error(f"⚠️ Unable to parse insights: {e}")
            return

    if insights:
        # Highlights
        if 'highlights' in insights and insights['highlights']:
            st.markdown("#### ✨ Key Highlights")
            for highlight in insights['highlights']:
                st.info(f"💡 {highlight}")

        # Background
        if 'background' in insights and insights['background']:
            st.markdown("#### 📖 Background")
            st.write(insights['background'])

        # Global trends
        if 'global_trends' in insights and insights['global_trends']:
            st.markdown("#### 🌍 Global Trends")
            st.write(insights['global_trends'])

        # Regional analysis
        if 'regional_analysis' in insights and insights['regional_analysis']:
            st.markdown("#### 🗺️ Regional Analysis")
            st.write(insights['regional_analysis'])

        # Explanatory notes
        if 'explanatory_notes' in insights and insights['explanatory_notes']:
            st.markdown("#### ⚠️ Explanatory Notes")
            st.write(insights['explanatory_notes'])

        # Generate report button
        st.markdown("---")
        if st.button("📄 Generate Full Report", type="primary"):
            generate_full_report()

def generate_full_report():
    """Generate a full analytical report."""
    
    # Check if we have all necessary data
    dataset_df = st.session_state.get('current_dataset')
    insights = st.session_state.get('current_insights')
    figures = st.session_state.get('current_figures', [])
    
    if not all([dataset_df is not None, insights]):
        st.error("❌ Missing data for report generation. Please run analysis and AI insights first.")
        return
    
    # Create analysis results object
    dataset_code = st.session_state.get('current_dataset_code')
    dataset_name = st.session_state.get('current_dataset_name', dataset_code)
    
    faostat_service = st.session_state.faostat_service
    summary_stats = faostat_service.generate_summary_stats(dataset_df)
    
    analysis_results = AnalysisResults(
        dataset_code=dataset_code,
        dataset_name=dataset_name,
        summary_stats=summary_stats,
        figures=figures,
        filter_options=AnalysisFilter(),  # Could store current filters
        insights=insights
    )
    
    # Generate report
    doc_service = st.session_state.get('doc_service')
    
    if doc_service:
        with st.spinner("📄 Generating professional report..."):
            try:
                # Default to PDF, but could let user choose
                report_path = doc_service.generate_brief(
                    analysis_results=analysis_results,
                    insights=insights,
                    format_type="pdf",
                    title=f"FAOSTAT {dataset_name} Analytical Brief"
                )
                
                st.success(f"✅ Report generated successfully!")
                st.info(f"📁 Saved to: {report_path}")
                
                # Offer download
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Report",
                        data=file.read(),
                        file_name=os.path.basename(report_path),
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                st.error(f"Error generating report: {str(e)}")
    else:
        st.error("❌ Document service not available.")