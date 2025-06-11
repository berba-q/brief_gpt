"""
Bookmarking system for saving and managing analysis configurations.

This module provides functionality to save, load, and manage analysis bookmarks,
allowing users to quickly return to specific analysis configurations.
"""

import streamlit as st
import json
import os
import logging
from datetime import datetime
from datetime import timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from models.data_models import Bookmark, AnalysisFilter
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BookmarkManager:
    """
    Manager for handling analysis bookmarks.
    
    Provides functionality to save, load, organize, and manage
    bookmarks for different analysis configurations.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the bookmark manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.bookmarks_dir = os.path.join(
            config_manager.get_output_directory(), 
            "bookmarks"
        )
        
        # Create bookmarks directory if it doesn't exist
        os.makedirs(self.bookmarks_dir, exist_ok=True)
        
        # Load existing bookmarks
        self.bookmarks = self.load_bookmarks()
        
        logger.info(f"Initialized bookmark manager with {len(self.bookmarks)} bookmarks")
    
    def save_bookmark(
        self,
        name: str,
        dataset_code: str,
        dataset_name: str,
        filters: AnalysisFilter,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a new bookmark.
        
        Args:
            name: Name for the bookmark
            dataset_code: Dataset code
            dataset_name: Dataset name
            filters: Analysis filters to save
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            str: Bookmark ID
        """
        
        # Generate unique ID
        bookmark_id = f"bookmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.bookmarks)}"
        
        # Create bookmark object
        bookmark = Bookmark(
            id=bookmark_id,
            name=name,
            dataset_code=dataset_code,
            dataset_name=dataset_name,
            filters=filters,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        # Add to bookmarks list
        self.bookmarks.append(bookmark)
        
        # Save to file
        self._save_bookmark_to_file(bookmark, tags or [])
        
        # Update session state
        st.session_state.bookmarks = self.bookmarks
        
        logger.info(f"Saved bookmark: {name}")
        return bookmark_id
    
    def load_bookmark(self, bookmark_id: str) -> Optional[Bookmark]:
        """
        Load a specific bookmark by ID.
        
        Args:
            bookmark_id: ID of the bookmark to load
            
        Returns:
            Bookmark object or None if not found
        """
        
        for bookmark in self.bookmarks:
            if bookmark.id == bookmark_id:
                return bookmark
        
        return None
    
    def delete_bookmark(self, bookmark_id: str) -> bool:
        """
        Delete a bookmark by ID.
        
        Args:
            bookmark_id: ID of the bookmark to delete
            
        Returns:
            bool: True if deleted successfully
        """
        
        # Find and remove from list
        for i, bookmark in enumerate(self.bookmarks):
            if bookmark.id == bookmark_id:
                # Remove from memory
                removed_bookmark = self.bookmarks.pop(i)
                
                # Remove file
                bookmark_file = os.path.join(self.bookmarks_dir, f"{bookmark_id}.json")
                if os.path.exists(bookmark_file):
                    os.remove(bookmark_file)
                
                # Update session state
                st.session_state.bookmarks = self.bookmarks
                
                logger.info(f"Deleted bookmark: {removed_bookmark.name}")
                return True
        
        return False
    
    def get_bookmarks(
        self,
        dataset_code: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search_term: Optional[str] = None
    ) -> List[Bookmark]:
        """
        Get bookmarks with optional filtering.
        
        Args:
            dataset_code: Filter by dataset code
            tags: Filter by tags
            search_term: Search in name and description
            
        Returns:
            List of matching bookmarks
        """
        
        filtered_bookmarks = self.bookmarks.copy()
        
        # Filter by dataset
        if dataset_code:
            filtered_bookmarks = [
                b for b in filtered_bookmarks 
                if b.dataset_code == dataset_code
            ]
        
        # Filter by search term
        if search_term:
            search_lower = search_term.lower()
            filtered_bookmarks = [
                b for b in filtered_bookmarks
                if (search_lower in b.name.lower() or 
                    (b.description and search_lower in b.description.lower()))
            ]
        
        # Sort by creation date (most recent first)
        filtered_bookmarks.sort(
            key=lambda x: datetime.fromisoformat(x.created_at),
            reverse=True
        )
        
        return filtered_bookmarks
    
    def load_bookmarks(self) -> List[Bookmark]:
        """Load all bookmarks from files."""
        
        bookmarks = []
        
        if not os.path.exists(self.bookmarks_dir):
            return bookmarks
        
        # Load all bookmark files
        for filename in os.listdir(self.bookmarks_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.bookmarks_dir, filename)
                    with open(filepath, 'r') as f:
                        bookmark_data = json.load(f)
                    
                    # Convert to Bookmark object
                    bookmark = self._dict_to_bookmark(bookmark_data['bookmark'])
                    bookmarks.append(bookmark)
                    
                except Exception as e:
                    logger.error(f"Error loading bookmark {filename}: {str(e)}")
        
        return bookmarks
    
    def _save_bookmark_to_file(self, bookmark: Bookmark, tags: List[str]):
        """Save bookmark to JSON file."""
        
        bookmark_data = {
            'bookmark': self._bookmark_to_dict(bookmark),
            'tags': tags,
            'saved_at': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.bookmarks_dir, f"{bookmark.id}.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(bookmark_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bookmark to file: {str(e)}")
    
    def _bookmark_to_dict(self, bookmark: Bookmark) -> Dict[str, Any]:
        """Convert bookmark to dictionary."""
        
        return {
            'id': bookmark.id,
            'name': bookmark.name,
            'dataset_code': bookmark.dataset_code,
            'dataset_name': bookmark.dataset_name,
            'created_at': bookmark.created_at,
            'description': bookmark.description,
            'filters': self._filters_to_dict(bookmark.filters) if bookmark.filters else None
        }
    
    def _dict_to_bookmark(self, data: Dict[str, Any]) -> Bookmark:
        """Convert dictionary to bookmark object."""
        
        filters = None
        if data.get('filters'):
            filters = self._dict_to_filters(data['filters'])
        
        return Bookmark(
            id=data['id'],
            name=data['name'],
            dataset_code=data['dataset_code'],
            dataset_name=data['dataset_name'],
            created_at=data['created_at'],
            description=data.get('description'),
            filters=filters
        )
    
    def _filters_to_dict(self, filters: AnalysisFilter) -> Dict[str, Any]:
        """Convert filters to dictionary."""
        
        return {
            'year_range': filters.year_range,
            'regions': filters.regions,
            'items': filters.items,
            'elements': filters.elements
        }
    
    def _dict_to_filters(self, data: Dict[str, Any]) -> AnalysisFilter:
        """Convert dictionary to filters object."""
        
        return AnalysisFilter(
            year_range=tuple(data['year_range']) if data.get('year_range') else None,
            regions=data.get('regions'),
            items=data.get('items'),
            elements=data.get('elements')
        )
    
    def export_bookmarks(self, filepath: str) -> bool:
        """
        Export all bookmarks to a file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            bool: True if successful
        """
        
        try:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_bookmarks': len(self.bookmarks),
                'bookmarks': [self._bookmark_to_dict(b) for b in self.bookmarks]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.bookmarks)} bookmarks to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting bookmarks: {str(e)}")
            return False
    
    def import_bookmarks(self, filepath: str) -> int:
        """
        Import bookmarks from a file.
        
        Args:
            filepath: Path to import file
            
        Returns:
            int: Number of bookmarks imported
        """
        
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for bookmark_data in import_data.get('bookmarks', []):
                try:
                    # Convert to bookmark object
                    bookmark = self._dict_to_bookmark(bookmark_data)
                    
                    # Check if bookmark already exists
                    if not any(b.id == bookmark.id for b in self.bookmarks):
                        self.bookmarks.append(bookmark)
                        # Save individual file
                        self._save_bookmark_to_file(bookmark, [])
                        imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error importing individual bookmark: {str(e)}")
            
            # Update session state
            st.session_state.bookmarks = self.bookmarks
            
            logger.info(f"Imported {imported_count} bookmarks from {filepath}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing bookmarks: {str(e)}")
            return 0


def render_bookmark_interface():
    """Render the bookmark management interface."""
    
    st.markdown("## 🔖 Bookmark Management")
    
    # Initialize bookmark manager
    if 'bookmark_manager' not in st.session_state:
        config_manager = st.session_state.get('config_manager')
        if config_manager:
            st.session_state.bookmark_manager = BookmarkManager(config_manager)
        else:
            st.error("Configuration manager not available")
            return
    
    bookmark_manager = st.session_state.bookmark_manager
    
    # Bookmark interface tabs
    tab1, tab2, tab3 = st.tabs(["📋 My Bookmarks", "➕ Save Current", "⚙️ Manage"])
    
    with tab1:
        render_bookmarks_list(bookmark_manager)
    
    with tab2:
        render_save_bookmark(bookmark_manager)
    
    with tab3:
        render_bookmark_management(bookmark_manager)


def render_bookmarks_list(bookmark_manager: BookmarkManager):
    """Render the list of saved bookmarks."""
    
    st.markdown("### 📚 Saved Bookmarks")
    
    # Search and filter controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search bookmarks",
            placeholder="Search by name or description...",
            key="bookmark_search"
        )
    
    with col2:
        # Get unique datasets for filtering
        all_datasets = list(set(b.dataset_code for b in bookmark_manager.bookmarks))
        dataset_filter = st.selectbox(
            "Filter by dataset",
            options=["All datasets"] + all_datasets,
            key="bookmark_dataset_filter"
        )
    
    # Apply filters
    dataset_code = None if dataset_filter == "All datasets" else dataset_filter
    bookmarks = bookmark_manager.get_bookmarks(
        dataset_code=dataset_code,
        search_term=search_term if search_term else None
    )
    
    # Display bookmarks
    if not bookmarks:
        st.info("📝 No bookmarks found. Save your first analysis to get started!")
        return
    
    st.info(f"📊 Showing {len(bookmarks)} bookmark(s)")
    
    # Display each bookmark
    for bookmark in bookmarks:
        render_bookmark_card(bookmark, bookmark_manager)


def render_bookmark_card(bookmark: Bookmark, bookmark_manager: BookmarkManager):
    """Render a single bookmark card."""
    
    with st.container():
        # Create card layout
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### 📊 {bookmark.name}")
            st.markdown(f"**Dataset:** {bookmark.dataset_name} (`{bookmark.dataset_code}`)")
            
            if bookmark.description:
                st.markdown(f"**Description:** {bookmark.description}")
            
            # Show filter summary
            if bookmark.filters:
                filter_summary = []
                
                if bookmark.filters.year_range:
                    filter_summary.append(f"Years: {bookmark.filters.year_range[0]}-{bookmark.filters.year_range[1]}")
                
                if bookmark.filters.regions:
                    region_count = len(bookmark.filters.regions)
                    filter_summary.append(f"Regions: {region_count} selected")
                
                if bookmark.filters.items:
                    item_count = len(bookmark.filters.items)
                    filter_summary.append(f"Items: {item_count} selected")
                
                if filter_summary:
                    st.markdown(f"**Filters:** {' | '.join(filter_summary)}")
            
            # Creation date
            created_date = datetime.fromisoformat(bookmark.created_at)
            st.caption(f"Created: {created_date.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            # Load bookmark button
            if st.button("📂 Load", key=f"load_{bookmark.id}"):
                load_bookmark_analysis(bookmark)
        
        with col3:
            # Delete bookmark button
            if st.button("🗑️ Delete", key=f"delete_{bookmark.id}"):
                if bookmark_manager.delete_bookmark(bookmark.id):
                    st.success(f"Deleted bookmark: {bookmark.name}")
                    st.rerun()
                else:
                    st.error("Failed to delete bookmark")
        
        st.markdown("---")


def render_save_bookmark(bookmark_manager: BookmarkManager):
    """Render the save bookmark interface."""
    
    st.markdown("### 💾 Save Current Analysis")
    
    # Check if there's an analysis to save
    current_dataset_code = st.session_state.get('current_dataset_code')
    current_dataset_name = st.session_state.get('current_dataset_name')
    
    if not current_dataset_code:
        st.warning("⚠️ No current analysis to save. Please load a dataset and configure analysis first.")
        return
    
    # Current analysis info
    st.info(f"📊 Current Analysis: **{current_dataset_name}** (`{current_dataset_code}`)")
    
    # Bookmark details form
    with st.form("save_bookmark_form"):
        bookmark_name = st.text_input(
            "Bookmark Name *",
            placeholder="e.g., Wheat Production Trends 2010-2020",
            help="Give your bookmark a descriptive name"
        )
        
        bookmark_description = st.text_area(
            "Description",
            placeholder="Optional description of what this analysis focuses on...",
            height=100
        )
        
        bookmark_tags = st.text_input(
            "Tags",
            placeholder="e.g., wheat, production, trends (comma-separated)",
            help="Add tags to organize your bookmarks"
        )
        
        # Extract current filters (this would need to be implemented based on your analysis state)
        current_filters = extract_current_filters()
        
        if current_filters:
            st.markdown("**Current Filters:**")
            display_filter_summary(current_filters)
        
        # Save button
        submitted = st.form_submit_button("💾 Save Bookmark", type="primary")
        
        if submitted and bookmark_name:
            # Parse tags
            tags = [tag.strip() for tag in bookmark_tags.split(',')] if bookmark_tags else []
            
            # Save bookmark
            try:
                bookmark_id = bookmark_manager.save_bookmark(
                    name=bookmark_name,
                    dataset_code=current_dataset_code,
                    dataset_name=current_dataset_name,
                    filters=current_filters,
                    description=bookmark_description if bookmark_description else None,
                    tags=tags
                )
                
                st.success(f"✅ Bookmark saved successfully! ID: {bookmark_id}")
                
            except Exception as e:
                st.error(f"❌ Error saving bookmark: {str(e)}")
        
        elif submitted and not bookmark_name:
            st.error("❌ Please provide a bookmark name")


def render_bookmark_management(bookmark_manager: BookmarkManager):
    """Render bookmark management tools."""
    
    st.markdown("### ⚙️ Bookmark Management")
    
    # Statistics
    total_bookmarks = len(bookmark_manager.bookmarks)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Bookmarks", total_bookmarks)
    
    with col2:
        # Count unique datasets
        unique_datasets = len(set(b.dataset_code for b in bookmark_manager.bookmarks))
        st.metric("Datasets Covered", unique_datasets)
    
    with col3:
        # Count recent bookmarks (last 7 days)
        recent_count = 0
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for bookmark in bookmark_manager.bookmarks:
            created_date = datetime.fromisoformat(bookmark.created_at)
            if created_date > cutoff_date:
                recent_count += 1
        
        st.metric("Recent (7 days)", recent_count)
    
    # Management actions
    st.markdown("#### 🔧 Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export/Import:**")
        
        # Export bookmarks
        if st.button("📤 Export All Bookmarks"):
            export_path = f"bookmarks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if bookmark_manager.export_bookmarks(export_path):
                st.success(f"✅ Exported to {export_path}")
                
                # Offer download
                with open(export_path, 'r') as f:
                    st.download_button(
                        label="📥 Download Export File",
                        data=f.read(),
                        file_name=export_path,
                        mime="application/json"
                    )
            else:
                st.error("❌ Export failed")
        
        # Import bookmarks
        uploaded_file = st.file_uploader(
            "📤 Import Bookmarks",
            type=['json'],
            help="Upload a previously exported bookmarks file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Import bookmarks
            imported_count = bookmark_manager.import_bookmarks(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            if imported_count > 0:
                st.success(f"✅ Imported {imported_count} bookmarks")
                st.rerun()
            else:
                st.warning("⚠️ No new bookmarks imported")
    
    with col2:
        st.markdown("**Maintenance:**")
        
        # Clear all bookmarks
        if st.button("🗑️ Clear All Bookmarks", type="secondary"):
            if st.button("⚠️ Confirm Clear All", type="secondary"):
                # Delete all bookmark files
                for bookmark in bookmark_manager.bookmarks:
                    bookmark_manager.delete_bookmark(bookmark.id)
                
                st.success("✅ All bookmarks cleared")
                st.rerun()
        
        # Cleanup orphaned files
        if st.button("🧹 Cleanup Files"):
            # This would clean up any orphaned bookmark files
            st.info("🧹 Cleanup completed")


def extract_current_filters() -> Optional[AnalysisFilter]:
    """Extract current analysis filters from session state."""
    
    # This would extract the current filters from your analysis state
    # Implementation depends on how you store analysis configuration
    
    # For now, return a basic filter structure
    # You'll need to implement this based on your actual analysis state management
    
    return AnalysisFilter(
        year_range=st.session_state.get('analysis_year_range'),
        regions=st.session_state.get('analysis_regions'),
        items=st.session_state.get('analysis_items'),
        elements=st.session_state.get('analysis_elements')
    )


def display_filter_summary(filters: AnalysisFilter):
    """Display a summary of analysis filters."""
    
    if filters.year_range:
        st.text(f"• Years: {filters.year_range[0]} - {filters.year_range[1]}")
    
    if filters.regions:
        region_text = f"{len(filters.regions)} regions" if len(filters.regions) > 3 else ", ".join(filters.regions)
        st.text(f"• Regions: {region_text}")
    
    if filters.items:
        item_text = f"{len(filters.items)} items" if len(filters.items) > 3 else ", ".join(filters.items)
        st.text(f"• Items: {item_text}")
    
    if filters.elements:
        element_text = f"{len(filters.elements)} elements" if len(filters.elements) > 3 else ", ".join(filters.elements)
        st.text(f"• Elements: {element_text}")


def load_bookmark_analysis(bookmark: Bookmark):
    """Load an analysis from a bookmark."""
    
    try:
        # Set current dataset
        st.session_state.current_dataset_code = bookmark.dataset_code
        st.session_state.current_dataset_name = bookmark.dataset_name
        
        # Apply filters if available
        if bookmark.filters:
            st.session_state.analysis_year_range = bookmark.filters.year_range
            st.session_state.analysis_regions = bookmark.filters.regions
            st.session_state.analysis_items = bookmark.filters.items
            st.session_state.analysis_elements = bookmark.filters.elements
        
        # Navigate to analysis page
        st.session_state.selected_page = "Analysis"
        
        st.success(f"✅ Loaded bookmark: {bookmark.name}")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error loading bookmark: {str(e)}")
        st.error(f"❌ Error loading bookmark: {str(e)}")
