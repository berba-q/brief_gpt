"""
Template manager for FAOSTAT Analytics Application.

This module provides a service class for managing thematic analysis templates,
including loading, saving, creating custom templates, and applying them to datasets.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from config.constants import THEMATIC_TEMPLATES, DEFAULT_BOOKMARKS_DIR
from models.data_models import ThematicTemplate, Bookmark, AnalysisFilter

# Set up logger
logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Manager for thematic analysis templates and bookmarks.
    
    This class provides methods to:
    - Load and manage predefined thematic templates
    - Create and save custom templates
    - Manage analysis bookmarks
    - Apply templates to generate analysis configurations
    
    Attributes:
        templates_dir (str): Directory for storing custom templates
        bookmarks_dir (str): Directory for storing bookmarks
        templates (dict): Dictionary of available templates
        bookmarks (dict): Dictionary of saved bookmarks
    """
    
    def __init__(
        self,
        templates_dir: str = "config/templates",
        bookmarks_dir: str = DEFAULT_BOOKMARKS_DIR
    ):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory for storing custom templates
            bookmarks_dir: Directory for storing bookmarks
        """
        self.templates_dir = templates_dir
        self.bookmarks_dir = bookmarks_dir
        
        # Initialize storage directories
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(bookmarks_dir, exist_ok=True)
        
        # Load templates and bookmarks
        self.templates = {}
        self.bookmarks = {}
        
        self._load_predefined_templates()
        self._load_custom_templates()
        self._load_bookmarks()
        
        logger.info(f"Initialized template manager with {len(self.templates)} templates and {len(self.bookmarks)} bookmarks")
    
    def _load_predefined_templates(self) -> None:
        """Load predefined thematic templates from constants."""
        try:
            for template_id, template_data in THEMATIC_TEMPLATES.items():
                template = ThematicTemplate(
                    id=template_id,
                    name=template_data["name"],
                    description=template_data["description"],
                    datasets=template_data["datasets"],
                    key_indicators=template_data["key_indicators"],
                    regions_focus=template_data["regions_focus"],
                    default_visualizations=template_data["default_visualizations"],
                    prompt_template=template_data["prompt_template"]
                )
                self.templates[template_id] = template
                
            logger.info(f"Loaded {len(THEMATIC_TEMPLATES)} predefined templates")
            
        except Exception as e:
            logger.error(f"Error loading predefined templates: {str(e)}")
    
    def _load_custom_templates(self) -> None:
        """Load custom templates from the templates directory."""
        try:
            if not os.path.exists(self.templates_dir):
                return
            
            for filename in os.listdir(self.templates_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.templates_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                        
                        template = ThematicTemplate(**template_data)
                        self.templates[template.id] = template
                        
                    except Exception as e:
                        logger.warning(f"Error loading template from {filename}: {str(e)}")
            
            custom_count = len([f for f in os.listdir(self.templates_dir) if f.endswith('.json')])
            if custom_count > 0:
                logger.info(f"Loaded {custom_count} custom templates")
                
        except Exception as e:
            logger.error(f"Error loading custom templates: {str(e)}")
    
    def _load_bookmarks(self) -> None:
        """Load bookmarks from the bookmarks directory."""
        try:
            if not os.path.exists(self.bookmarks_dir):
                return
            
            for filename in os.listdir(self.bookmarks_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.bookmarks_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            bookmark_data = json.load(f)
                        
                        # Convert filters dict back to AnalysisFilter object
                        if 'filters' in bookmark_data:
                            filters_data = bookmark_data['filters']
                            bookmark_data['filters'] = AnalysisFilter(**filters_data)
                        
                        bookmark = Bookmark(**bookmark_data)
                        self.bookmarks[bookmark.id] = bookmark
                        
                    except Exception as e:
                        logger.warning(f"Error loading bookmark from {filename}: {str(e)}")
            
            if len(self.bookmarks) > 0:
                logger.info(f"Loaded {len(self.bookmarks)} bookmarks")
                
        except Exception as e:
            logger.error(f"Error loading bookmarks: {str(e)}")
    
    def get_all_templates(self) -> Dict[str, ThematicTemplate]:
        """
        Get all available templates.
        
        Returns:
            Dictionary of template ID to ThematicTemplate objects
        """
        return self.templates.copy()
    
    def get_template(self, template_id: str) -> Optional[ThematicTemplate]:
        """
        Get a specific template by ID.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            ThematicTemplate object or None if not found
        """
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str = None) -> List[ThematicTemplate]:
        """
        Get templates by category or theme.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of ThematicTemplate objects
        """
        templates = list(self.templates.values())
        
        if category:
            # Filter by category (could be in name or description)
            category_lower = category.lower()
            templates = [
                t for t in templates 
                if category_lower in t.name.lower() or category_lower in t.description.lower()
            ]
        
        return templates
    
    def create_template(
        self,
        template_id: str,
        name: str,
        description: str,
        datasets: List[Dict[str, str]],
        key_indicators: List[str],
        regions_focus: List[str],
        default_visualizations: List[str],
        prompt_template: str,
        save_to_file: bool = True
    ) -> ThematicTemplate:
        """
        Create a new thematic template.
        
        Args:
            template_id: Unique identifier for the template
            name: Display name for the template
            description: Description of the template's purpose
            datasets: List of dataset dictionaries with 'code' and 'name'
            key_indicators: List of key indicators to focus on
            regions_focus: List of regions to focus analysis on
            default_visualizations: List of default visualization types
            prompt_template: Template for generating analysis prompts
            save_to_file: Whether to save the template to a file
            
        Returns:
            Created ThematicTemplate object
        """
        template = ThematicTemplate(
            id=template_id,
            name=name,
            description=description,
            datasets=datasets,
            key_indicators=key_indicators,
            regions_focus=regions_focus,
            default_visualizations=default_visualizations,
            prompt_template=prompt_template
        )
        
        # Add to memory
        self.templates[template_id] = template
        
        # Save to file if requested
        if save_to_file:
            self.save_template(template)
        
        logger.info(f"Created new template: {name}")
        return template
    
    def save_template(self, template: ThematicTemplate) -> bool:
        """
        Save a template to a file.
        
        Args:
            template: ThematicTemplate object to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            filename = f"template_{template.id}.json"
            filepath = os.path.join(self.templates_dir, filename)
            
            # Convert to dictionary
            template_dict = asdict(template)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved template {template.id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template {template.id}: {str(e)}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a custom template.
        
        Args:
            template_id: ID of the template to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Check if it's a predefined template (cannot be deleted)
            if template_id in THEMATIC_TEMPLATES:
                logger.warning(f"Cannot delete predefined template: {template_id}")
                return False
            
            # Remove from memory
            if template_id in self.templates:
                del self.templates[template_id]
            
            # Remove file
            filename = f"template_{template_id}.json"
            filepath = os.path.join(self.templates_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted template {template_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting template {template_id}: {str(e)}")
            return False
    
    def get_all_bookmarks(self) -> Dict[str, Bookmark]:
        """
        Get all saved bookmarks.
        
        Returns:
            Dictionary of bookmark ID to Bookmark objects
        """
        return self.bookmarks.copy()
    
    def get_bookmark(self, bookmark_id: str) -> Optional[Bookmark]:
        """
        Get a specific bookmark by ID.
        
        Args:
            bookmark_id: ID of the bookmark to retrieve
            
        Returns:
            Bookmark object or None if not found
        """
        return self.bookmarks.get(bookmark_id)
    
    def create_bookmark(
        self,
        bookmark_id: str,
        name: str,
        dataset_code: str,
        dataset_name: str,
        filters: AnalysisFilter,
        description: Optional[str] = None,
        save_to_file: bool = True
    ) -> Bookmark:
        """
        Create a new analysis bookmark.
        
        Args:
            bookmark_id: Unique identifier for the bookmark
            name: Display name for the bookmark
            dataset_code: Code of the dataset
            dataset_name: Name of the dataset
            filters: Analysis filters applied
            description: Optional description
            save_to_file: Whether to save the bookmark to a file
            
        Returns:
            Created Bookmark object
        """
        bookmark = Bookmark(
            id=bookmark_id,
            name=name,
            dataset_code=dataset_code,
            dataset_name=dataset_name,
            filters=filters,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        # Add to memory
        self.bookmarks[bookmark_id] = bookmark
        
        # Save to file if requested
        if save_to_file:
            self.save_bookmark(bookmark)
        
        logger.info(f"Created new bookmark: {name}")
        return bookmark
    
    def save_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Save a bookmark to a file.
        
        Args:
            bookmark: Bookmark object to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            filename = f"bookmark_{bookmark.id}.json"
            filepath = os.path.join(self.bookmarks_dir, filename)
            
            # Convert to dictionary
            bookmark_dict = asdict(bookmark)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(bookmark_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved bookmark {bookmark.id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving bookmark {bookmark.id}: {str(e)}")
            return False
    
    def delete_bookmark(self, bookmark_id: str) -> bool:
        """
        Delete a bookmark.
        
        Args:
            bookmark_id: ID of the bookmark to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Remove from memory
            if bookmark_id in self.bookmarks:
                del self.bookmarks[bookmark_id]
            
            # Remove file
            filename = f"bookmark_{bookmark_id}.json"
            filepath = os.path.join(self.bookmarks_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted bookmark {bookmark_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting bookmark {bookmark_id}: {str(e)}")
            return False
    
    def get_bookmarks_by_dataset(self, dataset_code: str) -> List[Bookmark]:
        """
        Get bookmarks for a specific dataset.
        
        Args:
            dataset_code: Code of the dataset
            
        Returns:
            List of Bookmark objects for the dataset
        """
        return [
            bookmark for bookmark in self.bookmarks.values()
            if bookmark.dataset_code == dataset_code
        ]
    
    def apply_template_to_dataset(
        self,
        template_id: str,
        dataset_code: str,
        available_regions: List[str],
        available_items: List[str],
        available_elements: List[str]
    ) -> Dict[str, Any]:
        """
        Apply a template to a dataset, generating analysis configuration.
        
        Args:
            template_id: ID of the template to apply
            dataset_code: Code of the dataset to analyze
            available_regions: List of available regions in the dataset
            available_items: List of available items in the dataset
            available_elements: List of available elements in the dataset
            
        Returns:
            Dictionary with analysis configuration
        """
        template = self.get_template(template_id)
        if not template:
            logger.error(f"Template {template_id} not found")
            return {}
        
        try:
            # Match template requirements with available data
            config = {
                "template_id": template_id,
                "template_name": template.name,
                "dataset_code": dataset_code,
                "analysis_focus": template.description,
                "suggested_regions": self._match_regions(template.regions_focus, available_regions),
                "suggested_items": self._match_items(template.key_indicators, available_items),
                "suggested_elements": self._match_elements(template.key_indicators, available_elements),
                "default_visualizations": template.default_visualizations,
                "prompt_template": template.prompt_template,
                "datasets_info": template.datasets
            }
            
            # Add specific recommendations
            config["recommendations"] = self._generate_recommendations(template, config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error applying template {template_id}: {str(e)}")
            return {}
    
    def _match_regions(self, template_regions: List[str], available_regions: List[str]) -> List[str]:
        """Match template region focus with available regions."""
        matched_regions = []
        
        for template_region in template_regions:
            if template_region in available_regions:
                matched_regions.append(template_region)
            else:
                # Try partial matching
                for available_region in available_regions:
                    if template_region.lower() in available_region.lower():
                        matched_regions.append(available_region)
                        break
        
        # If no matches found, return top regions by name
        if not matched_regions:
            matched_regions = available_regions[:10]
        
        return matched_regions
    
    def _match_items(self, key_indicators: List[str], available_items: List[str]) -> List[str]:
        """Match template key indicators with available items."""
        matched_items = []
        
        for indicator in key_indicators:
            for item in available_items:
                if indicator.lower() in item.lower() or item.lower() in indicator.lower():
                    matched_items.append(item)
        
        # Remove duplicates
        matched_items = list(set(matched_items))
        
        # If no matches found, return sample items
        if not matched_items:
            matched_items = available_items[:5]
        
        return matched_items
    
    def _match_elements(self, key_indicators: List[str], available_elements: List[str]) -> List[str]:
        """Match template key indicators with available elements."""
        matched_elements = []
        
        for indicator in key_indicators:
            for element in available_elements:
                if indicator.lower() in element.lower() or element.lower() in indicator.lower():
                    matched_elements.append(element)
        
        # Remove duplicates
        matched_elements = list(set(matched_elements))
        
        # If no matches found, return all elements
        if not matched_elements:
            matched_elements = available_elements
        
        return matched_elements
    
    def _generate_recommendations(self, template: ThematicTemplate, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific recommendations based on template and dataset."""
        recommendations = {
            "analysis_approach": f"Focus on {template.name.lower()} using the suggested regions and indicators.",
            "key_questions": [
                f"What are the main trends in {', '.join(config['suggested_elements'][:3])}?",
                f"How do {', '.join(config['suggested_regions'][:3])} compare in performance?",
                "What factors might explain the observed patterns?",
                "What policy implications can be drawn from the data?"
            ],
            "visualization_priority": config["default_visualizations"][:3],
            "suggested_time_range": "Use the most recent 10-15 years for trend analysis"
        }
        
        return recommendations
    
    def export_templates(self, filepath: str) -> bool:
        """
        Export all templates to a JSON file.
        
        Args:
            filepath: Path to save the exported templates
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "templates": {}
            }
            
            for template_id, template in self.templates.items():
                export_data["templates"][template_id] = asdict(template)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(self.templates)} templates to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting templates: {str(e)}")
            return False
    
    def import_templates(self, filepath: str) -> bool:
        """
        Import templates from a JSON file.
        
        Args:
            filepath: Path to the file containing templates
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if "templates" not in import_data:
                logger.error("Invalid template file format")
                return False
            
            imported_count = 0
            for template_id, template_data in import_data["templates"].items():
                try:
                    template = ThematicTemplate(**template_data)
                    self.templates[template_id] = template
                    self.save_template(template)
                    imported_count += 1
                except Exception as e:
                    logger.warning(f"Error importing template {template_id}: {str(e)}")
            
            logger.info(f"Imported {imported_count} templates from {filepath}")
            return imported_count > 0
            
        except Exception as e:
            logger.error(f"Error importing templates: {str(e)}")
            return False