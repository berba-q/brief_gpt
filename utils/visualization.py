"""
Visualization generator for FAOSTAT Analytics Application.

This module provides a class for generating standardized, high-quality
visualizations from FAOSTAT data for use in analytical briefs and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from io import BytesIO

from config.constants import (
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_DPI,
    DEFAULT_REGION_COLORS,
    DEFAULT_RANKING_COLORS
)

from models.data_models import VisualizationInfo

# Set up logger
logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generator for standardized data visualizations.
    
    This class provides methods to create various types of visualizations
    from FAOSTAT data, maintaining consistent styling and formatting across
    all outputs.
    
    Attributes:
        output_dir (str): Directory to save generated figures
        figure_width (float): Default figure width in inches
        figure_height (float): Default figure height in inches
        dpi (int): Default dots per inch for saved figures
        region_colors (dict): Color mapping for regions
        nutrient_colors (dict): Color mapping for nutrients
        ranking_colors (dict): Color mapping for rankings
    """
    
    def __init__(
        self,
        output_dir: str = "output/figures",
        figure_width: float = DEFAULT_FIGURE_WIDTH,
        figure_height: float = DEFAULT_FIGURE_HEIGHT,
        dpi: int = DEFAULT_DPI,
        region_colors: Optional[Dict[str, str]] = None,
        ranking_colors: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory to save generated figures
            figure_width: Default figure width in inches
            figure_height: Default figure height in inches
            dpi: Default dots per inch for saved figures
            region_colors: Color mapping for regions (or None for defaults)
            ranking_colors: Color mapping for rankings (or None for defaults)
        """
        self.output_dir = output_dir
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.dpi = dpi
        
        # Use provided color maps or defaults
        self.region_colors = region_colors or DEFAULT_REGION_COLORS
        self.ranking_colors = ranking_colors or DEFAULT_RANKING_COLORS
        
        # Dynamic color palettes for items and elements
        self.item_colors = {}  # Will be populated dynamically
        self.element_colors = {}  # Will be populated dynamically
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default Matplotlib/Seaborn style
        self._set_default_style()
        
        logger.info(f"Initialized visualization generator with output directory: {output_dir}")
    
    def _set_default_style(self) -> None:
        """Set default visualization style."""
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Set Matplotlib params
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.figsize': (self.figure_width, self.figure_height)
        })
    
    def _generate_color_palette(self, categories: List[str], palette_type: str = "items") -> Dict[str, str]:
        """
        Generate a dynamic color palette for categorical data.
        
        Args:
            categories: List of category names to assign colors to
            palette_type: Type of palette ("items" or "elements")
            
        Returns:
            Dictionary mapping category names to hex colors
        """
        import colorsys
        
        # Use different base hues for different palette types
        base_hues = {
            "items": 0.3,     # Start with green-ish for items
            "elements": 0.6   # Start with blue-ish for elements
        }
        
        base_hue = base_hues.get(palette_type, 0.0)
        num_categories = len(categories)
        
        if num_categories == 0:
            return {}
        
        color_map = {}
        
        # Generate evenly spaced colors around the color wheel
        for i, category in enumerate(categories):
            # Calculate hue with even spacing
            hue = (base_hue + (i / num_categories)) % 1.0
            
            # Use good saturation and value for visibility
            saturation = 0.7
            value = 0.9
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Convert to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            
            color_map[category] = hex_color
        
        return color_map
    
    def _get_color_for_category(self, category: str, category_type: str) -> str:
        """
        Get a color for a specific category, generating one if needed.
        
        Args:
            category: The category name
            category_type: Type of category ("area", "item", "element")
            
        Returns:
            Hex color string
        """
        # Check predefined color maps first
        if category_type == "area" and category in self.region_colors:
            return self.region_colors[category]
        
        # For items, check if we have a generated palette
        if category_type == "item":
            if category not in self.item_colors:
                # This shouldn't happen if we're calling _ensure_color_palettes first
                return "#3498db"  # Default blue
            return self.item_colors[category]
        
        # For elements, check if we have a generated palette
        if category_type == "element":
            if category not in self.element_colors:
                # This shouldn't happen if we're calling _ensure_color_palettes first
                return "#2ecc71"  # Default green
            return self.element_colors[category]
        
        # Default fallback
        return "#95a5a6"  # Gray
    
    def _ensure_color_palettes(self, df: pd.DataFrame) -> None:
        """
        Ensure color palettes are generated for the current dataset.
        
        Args:
            df: Dataset DataFrame
        """
        # Generate item colors if Items column exists
        if 'Item' in df.columns:
            unique_items = df['Item'].unique().tolist()
            if not self.item_colors or not all(item in self.item_colors for item in unique_items):
                self.item_colors = self._generate_color_palette(unique_items, "items")
        
        # Generate element colors if Elements column exists
        if 'Element' in df.columns:
            unique_elements = df['Element'].unique().tolist()
            if not self.element_colors or not all(element in self.element_colors for element in unique_elements):
                self.element_colors = self._generate_color_palette(unique_elements, "elements")
    
    def _get_timestamp(self) -> str:
        """Get a timestamp string for filename uniqueness."""
        return datetime.now().strftime('%Y%m%d%H%M%S')
        """Get a timestamp string for filename uniqueness."""
        return datetime.now().strftime('%Y%m%d%H%M%S')
    
    def _save_figure(self, plt_figure: plt.Figure, name: str) -> str:
        """
        Save a figure to disk.
        
        Args:
            plt_figure: Matplotlib figure object
            name: Base name for the file (without extension)
            
        Returns:
            str: Path to the saved figure
        """
        timestamp = self._get_timestamp()
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure tight layout
        plt_figure.tight_layout()
        
        # Save the figure
        plt_figure.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        
        return filepath
    
    def generate_visualizations(
        self, 
        df: pd.DataFrame, 
        dataset_name: str
    ) -> List[VisualizationInfo]:
        """
        Generate appropriate visualizations based on the dataset.
        
        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset for titles
            
        Returns:
            List of VisualizationInfo objects
        """
        if df is None or df.empty:
            logger.warning("Cannot generate visualizations for empty dataset")
            return []
        
        # Ensure color palettes are generated for this dataset
        self._ensure_color_palettes(df)
        
        figures = []
        
        # Check for common dimensions to decide what visualizations to create
        has_year = 'Year' in df.columns
        has_area = 'Area' in df.columns
        has_item = 'Item' in df.columns
        has_element = 'Element' in df.columns
        has_value = 'Value' in df.columns
        
        # Only proceed if we have values to plot
        if not has_value:
            logger.warning("Dataset missing 'Value' column for visualizations")
            return figures
        
        # Generate time series visualization if time data is available
        if has_year:
            try:
                time_series_fig = self.create_time_series(df, dataset_name)
                if time_series_fig:
                    figures.append(time_series_fig)
            except Exception as e:
                logger.error(f"Error creating time series: {str(e)}")
        
        # Generate regional comparison if area data is available
        if has_area:
            try:
                regional_fig = self.create_regional_comparison(df, dataset_name)
                if regional_fig:
                    figures.append(regional_fig)
            except Exception as e:
                logger.error(f"Error creating regional comparison: {str(e)}")
        
        # Generate item comparison if item data is available
        if has_item:
            try:
                item_fig = self.create_item_comparison(df, dataset_name)
                if item_fig:
                    figures.append(item_fig)
            except Exception as e:
                logger.error(f"Error creating item comparison: {str(e)}")
        
        # Generate element comparison if element data is available
        if has_element and df['Element'].nunique() > 1:
            try:
                element_fig = self.create_element_comparison(df, dataset_name)
                if element_fig:
                    figures.append(element_fig)
            except Exception as e:
                logger.error(f"Error creating element comparison: {str(e)}")
        
        # Generate correlation heatmap if multiple numeric columns exist
        if df.select_dtypes(include=['number']).columns.size > 2:
            try:
                correlation_fig = self.create_correlation_heatmap(df, dataset_name)
                if correlation_fig:
                    figures.append(correlation_fig)
            except Exception as e:
                logger.error(f"Error creating correlation heatmap: {str(e)}")
        
        # Generate distribution chart if value column exists
        if has_value:
            try:
                distribution_fig = self.create_distribution_chart(df, dataset_name)
                if distribution_fig:
                    figures.append(distribution_fig)
            except Exception as e:
                logger.error(f"Error creating distribution chart: {str(e)}")
        
        # Generate treemap for hierarchical data if multiple categorical dimensions exist
        if sum([has_area, has_item, has_element]) >= 2:
            try:
                treemap_fig = self.create_treemap(df, dataset_name)
                if treemap_fig:
                    figures.append(treemap_fig)
            except Exception as e:
                logger.error(f"Error creating treemap: {str(e)}")
        
        return figures
    
    def create_time_series(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create a time series visualization.
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Year' not in df.columns or 'Value' not in df.columns:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Determine which dimension to group by based on cardinality
        group_by = None
        group_candidates = []
        
        if 'Item' in df.columns and df['Item'].nunique() <= 10:
            group_candidates.append(('Item', df['Item'].nunique()))
        
        if 'Element' in df.columns and df['Element'].nunique() <= 10:
            group_candidates.append(('Element', df['Element'].nunique()))
        
        if 'Area' in df.columns and df['Area'].nunique() <= 10:
            group_candidates.append(('Area', df['Area'].nunique()))
        
        # Choose the dimension with the most categories (but still <= 10)
        if group_candidates:
            group_candidates.sort(key=lambda x: x[1], reverse=True)
            group_by = group_candidates[0][0]
        
        # Determine appropriate year grouping to avoid overcrowding
        year_count = df['Year'].nunique()
        year_aggregation = None
        
        if year_count > 30:
            # Too many years, group by 5-year periods
            df['Year_Group'] = (df['Year'] // 5) * 5
            x_column = 'Year_Group'
            year_aggregation = "5-year periods"
        else:
            x_column = 'Year'
        
        # Generate the plot
        if group_by:
            # Group by the selected dimension
            for name, group in df.groupby(group_by):
                # Aggregate by year or year group
                agg_data = group.groupby(x_column)['Value'].sum().reset_index()
                agg_data = agg_data.sort_values(x_column)
                
                # Plot the line
                color = self._get_color_for_category(name, group_by.lower())
                
                ax.plot(
                    agg_data[x_column], 
                    agg_data['Value'], 
                    marker='o', 
                    linewidth=2, 
                    label=name,
                    color=color
                )
            
            # Add legend
            ax.legend(title=group_by)
            
        else:
            # No grouping dimension, aggregate all by year
            agg_data = df.groupby(x_column)['Value'].sum().reset_index()
            agg_data = agg_data.sort_values(x_column)
            
            ax.plot(
                agg_data[x_column], 
                agg_data['Value'], 
                marker='o', 
                linewidth=2,
                color='#3498db'  # Default blue
            )
        
        # Format axes with appropriate scale
        max_value = df['Value'].max()
        if max_value >= 1_000_000:
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1_000_000:.1f}M')
            y_label = 'Value (millions)'
        elif max_value >= 1_000:
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1_000:.1f}K')
            y_label = 'Value (thousands)'
        else:
            y_label = 'Value'
        
        # Set labels and title
        ax.set_xlabel('Year' if not year_aggregation else f'Year ({year_aggregation})')
        ax.set_ylabel(y_label)
        
        if 'Element' in df.columns and df['Element'].nunique() == 1:
            element = df['Element'].iloc[0]
            title = f"{title_prefix}: {element} over time"
        else:
            title = f"{title_prefix}: Trends over time"
        
        ax.set_title(title)
        
        # Customize appearance
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Save figure
        filename = self._save_figure(fig, "time_series")
        plt.close(fig)
        
        # Year range for description
        year_min = int(df['Year'].min())
        year_max = int(df['Year'].max())
        
        return VisualizationInfo(
            type="time_series",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Time series showing trends in {title_prefix} from {year_min} to {year_max}."
        )
    
    def create_regional_comparison(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create a regional comparison visualization.
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Area' not in df.columns or 'Value' not in df.columns:
            return None
        
        # Limit to top areas for readability
        area_totals = df.groupby('Area')['Value'].sum().reset_index()
        
        # Determine how many areas to show based on the dataset
        num_areas = min(10, area_totals['Area'].nunique())
        top_areas = area_totals.nlargest(num_areas, 'Value')
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Generate the plot
        if 'Year' in df.columns and df['Year'].nunique() > 1:
            # Use the most recent year and an earlier comparable year for comparison
            latest_year = df['Year'].max()
            
            # Find an appropriate comparison year
            year_options = sorted(df['Year'].unique())
            if len(year_options) >= 2:
                earlier_idx = max(0, len(year_options) - 6)  # 5 years before latest if available
                earlier_year = year_options[earlier_idx]
            else:
                earlier_year = latest_year
            
            # Filter for the two years and top areas
            comparison_df = df[df['Year'].isin([earlier_year, latest_year]) & 
                              df['Area'].isin(top_areas['Area'])]
            
            # Pivot and plot
            pivot_df = comparison_df.pivot_table(
                index='Area', 
                columns='Year', 
                values='Value', 
                aggfunc='sum'
            ).reset_index()
            
            # Sort by the latest year values
            pivot_df = pivot_df.sort_values(by=latest_year, ascending=True)
            
            # Plot horizontal bars
            bars1 = ax.barh(
                pivot_df['Area'], 
                pivot_df[latest_year], 
                label=str(latest_year),
                color='#3498db'  # Blue
            )
            
            bars2 = ax.barh(
                pivot_df['Area'], 
                pivot_df[earlier_year], 
                label=str(earlier_year), 
                alpha=0.6,
                color='#e74c3c'  # Red
            )
            
            # Add data labels
            for bar in bars1:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                        va='center', ha='left', fontsize=8)
            
            ax.legend()
            title = f"{title_prefix}: Regional Comparison ({earlier_year} vs {latest_year})"
            
        else:
            # Simple bar chart of regional totals
            top_areas = top_areas.sort_values(by='Value', ascending=True)
            
            # Use region colors where available, otherwise generate colors
            colors = [self._get_color_for_category(area, "area") for area in top_areas['Area']]
            
            bars = ax.barh(top_areas['Area'], top_areas['Value'], color=colors)
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width * 1.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                        va='center', ha='left', fontsize=8)
            
            title = f"{title_prefix}: Regional Comparison"
        
        # Format axes with appropriate scale
        max_value = max(df.groupby('Area')['Value'].sum())
        if max_value >= 1_000_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000_000:.1f}M')
            x_label = 'Value (millions)'
        elif max_value >= 1_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000:.1f}K')
            x_label = 'Value (thousands)'
        else:
            x_label = 'Value'
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel('Region/Country')
        ax.set_title(title)
        
        # Customize appearance
        ax.grid(True, alpha=0.3, axis='x')
        
        # Save figure
        filename = self._save_figure(fig, "regional_comparison")
        plt.close(fig)
        
        return VisualizationInfo(
            type="regional_comparison",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Comparison of {title_prefix} across regions and countries."
        )
    
    def create_item_comparison(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create an item comparison visualization.
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Item' not in df.columns or 'Value' not in df.columns:
            return None
        
        # Limit to top items for readability
        item_totals = df.groupby('Item')['Value'].sum().reset_index()
        
        # Determine how many items to show based on the dataset
        num_items = min(10, item_totals['Item'].nunique())
        top_items = item_totals.nlargest(num_items, 'Value')
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Sort and plot
        top_items = top_items.sort_values(by='Value', ascending=True)
        
        # Use dynamic item colors
        colors = [self._get_color_for_category(item, "item") for item in top_items['Item']]
        
        bars = ax.barh(top_items['Item'], top_items['Value'], color=colors)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                    va='center', ha='left', fontsize=8)
        
        # Format axes with appropriate scale
        max_value = max(df.groupby('Item')['Value'].sum())
        if max_value >= 1_000_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000_000:.1f}M')
            x_label = 'Value (millions)'
        elif max_value >= 1_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000:.1f}K')
            x_label = 'Value (thousands)'
        else:
            x_label = 'Value'
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel('Item')
        title = f"{title_prefix}: Item Comparison"
        ax.set_title(title)
        
        # Customize appearance
        ax.grid(True, alpha=0.3, axis='x')
        
        # Save figure
        filename = self._save_figure(fig, "item_comparison")
        plt.close(fig)
        
        return VisualizationInfo(
            type="item_comparison",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Comparison of different items in {title_prefix}."
        )
    
    def create_element_comparison(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create an element comparison visualization (pie chart).
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Element' not in df.columns or 'Value' not in df.columns:
            return None
        
        # Get element totals
        element_totals = df.groupby('Element')['Value'].sum().reset_index()
        
        # Only create visualization if we have multiple elements
        if len(element_totals) <= 1:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Sort by value
        element_totals = element_totals.sort_values('Value', ascending=False)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            element_totals['Value'], 
            labels=element_totals['Element'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )
        
        # Customize appearance
        plt.setp(autotexts, size=10, weight='bold')
        ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        
        # Set title
        title = f"{title_prefix}: Element Distribution"
        ax.set_title(title)
        
        # Save figure
        filename = self._save_figure(fig, "element_distribution")
        plt.close(fig)
        
        return VisualizationInfo(
            type="element_distribution",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Distribution of different elements in {title_prefix}."
        )
    
    def create_correlation_heatmap(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create a correlation heatmap visualization.
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove ID columns (usually not interesting for correlation)
        exclude_patterns = ['code', 'id', 'flag']
        numeric_cols = [col for col in numeric_cols if not any(pat in col.lower() for pat in exclude_patterns)]
        
        # Need at least 3 numeric columns for an interesting heatmap
        if len(numeric_cols) < 3:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=0.5,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        
        # Set title
        title = f"{title_prefix}: Correlation Matrix"
        ax.set_title(title)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Save figure
        filename = self._save_figure(fig, "correlation_heatmap")
        plt.close(fig)
        
        return VisualizationInfo(
            type="correlation_heatmap",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Correlation heatmap showing relationships between numeric variables in {title_prefix}."
        )
    
    def create_distribution_chart(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create a distribution chart (histogram with KDE).
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Value' not in df.columns:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Create histogram with KDE
        sns.histplot(
            data=df,
            x='Value',
            kde=True,
            ax=ax,
            color='#3498db'
        )
        
        # Add vertical line for mean
        mean_value = df['Value'].mean()
        ax.axvline(
            x=mean_value,
            color='red',
            linestyle='--',
            label=f'Mean: {mean_value:.2f}'
        )
        
        # Add vertical line for median
        median_value = df['Value'].median()
        ax.axvline(
            x=median_value,
            color='green',
            linestyle='-.',
            label=f'Median: {median_value:.2f}'
        )
        
        # Set labels and title
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        title = f"{title_prefix}: Distribution of Values"
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Customize appearance
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = self._save_figure(fig, "distribution_chart")
        plt.close(fig)
        
        return VisualizationInfo(
            type="distribution_chart",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Distribution of values in {title_prefix}, showing the spread and central tendency."
        )
    
    def create_treemap(
        self, 
        df: pd.DataFrame, 
        title_prefix: str
    ) -> Optional[VisualizationInfo]:
        """
        Create a treemap visualization for hierarchical data.
        
        Args:
            df: Dataset DataFrame
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if 'Value' not in df.columns:
            return None
        
        # Need at least 2 categorical columns for hierarchy
        categorical_cols = []
        if 'Area' in df.columns:
            categorical_cols.append('Area')
        if 'Item' in df.columns:
            categorical_cols.append('Item')
        if 'Element' in df.columns:
            categorical_cols.append('Element')
        
        if len(categorical_cols) < 2:
            return None
        
        # Select top 2 categorical columns with lowest cardinality for better visualization
        cat_counts = [(col, df[col].nunique()) for col in categorical_cols]
        cat_counts.sort(key=lambda x: x[1])
        selected_cats = [cat[0] for cat in cat_counts[:2]]
        
        # Create figure using matplotlib's treemap functionality
        try:
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
            
            # Aggregate data by the selected categories
            agg_data = df.groupby(selected_cats)['Value'].sum().reset_index()
            
            # Normalize values for rectangle sizes
            total_value = agg_data['Value'].sum()
            agg_data['norm_value'] = agg_data['Value'] / total_value
            
            # Sort by normalized value descending
            agg_data = agg_data.sort_values('norm_value', ascending=False)
            
            # Create treemap layout (simplified algorithm)
            rectangles = []
            labels = []
            colors = []
            
            # Create a simple grid layout
            n_rects = len(agg_data)
            grid_size = int(np.ceil(np.sqrt(n_rects)))
            
            for i, (_, row) in enumerate(agg_data.iterrows()):
                # Calculate position
                x = (i % grid_size) / grid_size
                y = (i // grid_size) / grid_size
                
                # Calculate size based on normalized value
                # Add small minimum size to ensure visibility
                width = max(0.05, row['norm_value'] * 0.5) / grid_size
                height = max(0.05, row['norm_value'] * 0.5) / grid_size
                
                # Create rectangle
                rect = Rectangle((x, y), width, height)
                rectangles.append(rect)
                
                # Create label
                label = f"{row[selected_cats[0]]}\n{row[selected_cats[1]]}"
                labels.append((x + width/2, y + height/2, label))
                
                # Assign color based on first category
                if selected_cats[0] == 'Area' and row[selected_cats[0]] in self.region_colors:
                    color = self.region_colors[row[selected_cats[0]]]
                elif selected_cats[0] == 'Item' and row[selected_cats[0]] in self.nutrient_colors:
                    color = self.nutrient_colors[row[selected_cats[0]]]
                else:
                    # Generate a color based on index
                    import colorsys
                    h = (i / n_rects) % 1.0
                    s = 0.7
                    v = 0.9
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    color = (r, g, b)
                
                colors.append(color)
            
            # Add rectangles to the plot
            collection = PatchCollection(rectangles, alpha=0.8)
            collection.set_facecolor(colors)
            ax.add_collection(collection)
            
            # Add labels
            for x, y, label in labels:
                ax.text(
                    x, y, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8
                )
            
            # Set limits and remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set title
            title = f"{title_prefix}: Treemap by {selected_cats[0]} and {selected_cats[1]}"
            ax.set_title(title)
            
            # Save figure
            filename = self._save_figure(fig, "treemap")
            plt.close(fig)
            
            return VisualizationInfo(
                type="treemap",
                title=title,
                filename=os.path.basename(filename),
                filepath=filename,
                description=f"Treemap showing hierarchical breakdown of {title_prefix} by {selected_cats[0]} and {selected_cats[1]}."
            )
            
        except Exception as e:
            logger.error(f"Error creating treemap: {str(e)}")
            return None
    
    def create_top_bottom_comparison(
        self, 
        top_entities: pd.DataFrame,
        bottom_entities: pd.DataFrame,
        entity_column: str,
        value_column: str = 'Value',
        title_prefix: str = "Comparison"
    ) -> Optional[VisualizationInfo]:
        """
        Create a comparison of top and bottom performers.
        
        Args:
            top_entities: DataFrame with top entities
            bottom_entities: DataFrame with bottom entities
            entity_column: Column containing entity names
            value_column: Column containing values
            title_prefix: Prefix for the chart title
            
        Returns:
            VisualizationInfo or None if creation fails
        """
        if entity_column not in top_entities.columns or value_column not in top_entities.columns:
            return None
        
        if entity_column not in bottom_entities.columns or value_column not in bottom_entities.columns:
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Prepare data for combined chart
        # For top entities, sort by value (descending)
        top_df = top_entities.sort_values(by=value_column, ascending=False).copy()
        top_df['Category'] = 'Top'
        
        # For bottom entities, sort by value (ascending)
        bottom_df = bottom_entities.sort_values(by=value_column, ascending=True).copy()
        bottom_df['Category'] = 'Bottom'
        
        # Combine data
        combined_df = pd.concat([top_df, bottom_df])
        
        # Plot as grouped bars
        combined_df['Abs_Value'] = combined_df[value_column].abs()  # For sorting
        
        # Sort the combined DataFrame by category (Top first) and absolute value
        combined_df = combined_df.sort_values(by=['Category', 'Abs_Value'], ascending=[True, True])
        
        # Use custom colors for top and bottom
        colors = {
            'Top': self.ranking_colors['top'],
            'Bottom': self.ranking_colors['bottom']
        }
        
        # Plot the grouped bar chart
        sns.barplot(
            x=value_column,
            y=entity_column,
            hue='Category',
            data=combined_df,
            palette=colors,
            ax=ax
        )
        
        # Format axes with appropriate scale
        max_value = combined_df[value_column].max()
        if max_value >= 1_000_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000_000:.1f}M')
            x_label = f'{value_column} (millions)'
        elif max_value >= 1_000:
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1_000:.1f}K')
            x_label = f'{value_column} (thousands)'
        else:
            x_label = value_column
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(entity_column)
        title = f"{title_prefix}: Top vs Bottom Comparison"
        ax.set_title(title)
        
        # Add text for ratio between top and bottom
        top_avg = top_entities[value_column].mean()
        bottom_avg = bottom_entities[value_column].mean()
        ratio = top_avg / max(bottom_avg, 0.001)  # Avoid division by zero
        
        plt.figtext(
            0.5, 0.01, 
            f"The average for top entities ({top_avg:.1f}) is {ratio:.1f}x higher than bottom entities ({bottom_avg:.1f})",
            ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
        )
        
        # Customize appearance
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom
        
        # Save figure
        filename = self._save_figure(fig, "top_bottom_comparison")
        plt.close(fig)
        
        return VisualizationInfo(
            type="top_vs_bottom",
            title=title,
            filename=os.path.basename(filename),
            filepath=filename,
            description=f"Comparison between top and bottom performing {entity_column}s in {title_prefix}."
        )