"""
Visualization gallery component for displaying and managing visualizations.

This module provides components for displaying collections of visualizations
in an organized gallery format with filtering and interaction capabilities.
"""

import streamlit as st
import os
import logging
from typing import List, Dict, Any, Optional
from models.data_models import VisualizationInfo

logger = logging.getLogger(__name__)

def render_visualization_gallery(
    figures: List[VisualizationInfo],
    title: str = "📊 Visualization Gallery",
    columns: int = 2,
    show_download: bool = True,
    show_details: bool = True
):
    """
    Render a gallery of visualizations.
    
    Args:
        figures: List of visualization info objects
        title: Gallery title
        columns: Number of columns in the gallery
        show_download: Whether to show download buttons
        show_details: Whether to show detailed information
    """
    
    if not figures:
        st.info("📊 No visualizations available yet. Generate some visualizations first!")
        return
    
    st.markdown(f"## {title}")
    st.markdown(f"*{len(figures)} visualization(s) available*")
    
    # Gallery controls
    render_gallery_controls(figures)
    
    # Filter figures based on controls
    filtered_figures = apply_gallery_filters(figures)
    
    # Render gallery grid
    render_gallery_grid(filtered_figures, columns, show_download, show_details)

def render_gallery_controls(figures: List[VisualizationInfo]):
    """Render gallery control options."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by visualization type
        available_types = list(set(fig.type for fig in figures))
        selected_types = st.multiselect(
            "Filter by type",
            options=available_types,
            default=available_types,
            key="gallery_type_filter"
        )
    
    with col2:
        # Sort options
        sort_option = st.selectbox(
            "Sort by",
            options=["title", "type", "newest", "oldest"],
            key="gallery_sort"
        )
    
    with col3:
        # View options
        view_mode = st.radio(
            "View mode",
            options=["grid", "list"],
            horizontal=True,
            key="gallery_view_mode"
        )

def apply_gallery_filters(figures: List[VisualizationInfo]) -> List[VisualizationInfo]:
    """Apply filters to the figure list."""
    
    filtered_figures = figures.copy()
    
    # Apply type filter
    selected_types = st.session_state.get('gallery_type_filter', [])
    if selected_types:
        filtered_figures = [fig for fig in filtered_figures if fig.type in selected_types]
    
    # Apply sorting
    sort_option = st.session_state.get('gallery_sort', 'title')
    
    if sort_option == "title":
        filtered_figures.sort(key=lambda x: x.title)
    elif sort_option == "type":
        filtered_figures.sort(key=lambda x: x.type)
    elif sort_option == "newest":
        filtered_figures.reverse()  # Assume figures are in creation order
    elif sort_option == "oldest":
        pass  # Keep original order
    
    return filtered_figures

def render_gallery_grid(
    figures: List[VisualizationInfo],
    columns: int,
    show_download: bool,
    show_details: bool
):
    """Render figures in a grid layout."""
    
    view_mode = st.session_state.get('gallery_view_mode', 'grid')
    
    if view_mode == "grid":
        render_grid_view(figures, columns, show_download, show_details)
    else:
        render_list_view(figures, show_download, show_details)

def render_grid_view(
    figures: List[VisualizationInfo],
    columns: int,
    show_download: bool,
    show_details: bool
):
    """Render figures in grid view."""
    
    # Create columns
    cols = st.columns(columns)
    
    for i, figure in enumerate(figures):
        col_idx = i % columns
        
        with cols[col_idx]:
            render_figure_card(figure, show_download, show_details)

def render_list_view(
    figures: List[VisualizationInfo],
    show_download: bool,
    show_details: bool
):
    """Render figures in list view."""
    
    for figure in figures:
        render_figure_card(figure, show_download, show_details, list_view=True)
        st.markdown("---")

def render_figure_card(
    figure: VisualizationInfo,
    show_download: bool = True,
    show_details: bool = True,
    list_view: bool = False
):
    """
    Render a single figure card.
    
    Args:
        figure: Visualization info object
        show_download: Whether to show download button
        show_details: Whether to show detailed information
        list_view: Whether this is being rendered in list view
    """
    
    # Figure container
    with st.container():
        if list_view:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                render_figure_image(figure)
            
            with col2:
                render_figure_info(figure, show_download, show_details)
        else:
            # Grid view - stack vertically
            render_figure_image(figure)
            render_figure_info(figure, show_download, show_details)

def render_figure_image(figure: VisualizationInfo):
    """Render the figure image."""
    
    if figure.filepath and os.path.exists(figure.filepath):
        try:
            st.image(
                figure.filepath,
                caption=figure.title,
                use_column_width=True
            )
        except Exception as e:
            logger.error(f"Error displaying image {figure.filepath}: {str(e)}")
            st.error(f"Could not display image: {figure.filename}")
    else:
        st.error(f"Image not found: {figure.filename}")

def render_figure_info(
    figure: VisualizationInfo,
    show_download: bool,
    show_details: bool
):
    """Render figure information and controls."""
    
    # Title and type
    st.markdown(f"**{figure.title}**")
    st.caption(f"Type: {figure.type}")
    
    # Description
    if show_details and figure.description:
        with st.expander("📝 Description"):
            st.write(figure.description)
    
    # Controls
    if show_download and figure.filepath and os.path.exists(figure.filepath):
        with open(figure.filepath, "rb") as file:
            st.download_button(
                label="💾 Download",
                data=file.read(),
                file_name=figure.filename,
                mime="image/png",
                key=f"download_{figure.filename}",
                use_container_width=True
            )
    
    # Additional actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 View Full", key=f"view_{figure.filename}"):
            show_figure_modal(figure)
    
    with col2:
        if st.button("📋 Copy Path", key=f"copy_{figure.filename}"):
            st.code(figure.filepath)

def show_figure_modal(figure: VisualizationInfo):
    """Show figure in a modal/expanded view."""
    
    st.session_state.modal_figure = figure
    
    # Create a modal-like experience
    st.markdown("---")
    st.markdown(f"## 🖼️ {figure.title}")
    
    if figure.filepath and os.path.exists(figure.filepath):
        st.image(figure.filepath, use_column_width=True)
        
        # Full details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Details:**")
            st.text(f"Type: {figure.type}")
            st.text(f"Filename: {figure.filename}")
            if figure.data_source:
                st.text(f"Source: {figure.data_source}")
        
        with col2:
            st.markdown("**Description:**")
            st.write(figure.description)
        
        # Close modal
        if st.button("❌ Close"):
            if 'modal_figure' in st.session_state:
                del st.session_state.modal_figure
            st.rerun()

def render_visualization_summary(figures: List[VisualizationInfo]):
    """Render a summary of available visualizations."""
    
    if not figures:
        return
    
    st.markdown("### 📊 Visualization Summary")
    
    # Count by type
    type_counts = {}
    for figure in figures:
        type_counts[figure.type] = type_counts.get(figure.type, 0) + 1
    
    # Display metrics
    cols = st.columns(min(4, len(type_counts)))
    
    for i, (viz_type, count) in enumerate(type_counts.items()):
        with cols[i % len(cols)]:
            st.metric(
                label=viz_type.replace("_", " ").title(),
                value=count
            )

def render_visualization_actions(figures: List[VisualizationInfo]):
    """Render bulk actions for visualizations."""
    
    if not figures:
        return
    
    st.markdown("### ⚡ Bulk Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Download All"):
            download_all_figures(figures)
    
    with col2:
        if st.button("🗑️ Clear Gallery"):
            clear_gallery()
    
    with col3:
        if st.button("🔄 Refresh Gallery"):
            refresh_gallery()

def download_all_figures(figures: List[VisualizationInfo]):
    """Create a downloadable archive of all figures."""
    
    try:
        import zipfile
        import io
        from datetime import datetime
        
        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for figure in figures:
                if figure.filepath and os.path.exists(figure.filepath):
                    zip_file.write(figure.filepath, figure.filename)
        
        zip_buffer.seek(0)
        
        # Offer download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 Download ZIP Archive",
            data=zip_buffer.getvalue(),
            file_name=f"faostat_visualizations_{timestamp}.zip",
            mime="application/zip"
        )
        
    except ImportError:
        st.error("ZIP functionality not available")
    except Exception as e:
        logger.error(f"Error creating ZIP archive: {str(e)}")
        st.error(f"Error creating archive: {str(e)}")

def clear_gallery():
    """Clear the visualization gallery."""
    
    if st.button("⚠️ Confirm Clear Gallery", type="secondary"):
        st.session_state.current_figures = []
        st.success("Gallery cleared!")
        st.rerun()

def refresh_gallery():
    """Refresh the gallery by checking for new files."""
    
    st.info("Gallery refreshed! Any new visualizations will appear here.")

def render_comparison_view(figures: List[VisualizationInfo]):
    """Render a comparison view of multiple figures."""
    
    if len(figures) < 2:
        st.info("Need at least 2 visualizations for comparison view")
        return
    
    st.markdown("### 🔄 Compare Visualizations")
    
    # Select figures to compare
    figure_options = {f"{fig.title} ({fig.type})": fig for fig in figures}
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1_key = st.selectbox(
            "Select first visualization",
            options=list(figure_options.keys()),
            key="compare_fig1"
        )
    
    with col2:
        fig2_key = st.selectbox(
            "Select second visualization",
            options=list(figure_options.keys()),
            key="compare_fig2"
        )
    
    # Display comparison
    if fig1_key and fig2_key and fig1_key != fig2_key:
        fig1 = figure_options[fig1_key]
        fig2 = figure_options[fig2_key]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {fig1.title}")
            if fig1.filepath and os.path.exists(fig1.filepath):
                st.image(fig1.filepath, use_column_width=True)
        
        with col2:
            st.markdown(f"#### {fig2.title}")
            if fig2.filepath and os.path.exists(fig2.filepath):
                st.image(fig2.filepath, use_column_width=True)