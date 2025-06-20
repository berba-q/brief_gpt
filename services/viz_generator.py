"""
Enhanced Visualization Generator for analytical briefs.

This service creates professional charts suitable for FAO analytical briefs:
- Trends charts (time series)
- Rankings charts (bar charts)
- Composition charts (pie/stacked bar)
"""

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Enhanced visualization generator for analytical briefs."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib parameters for professional output
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        # Professional color palette
        self.colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
            '#4A5568', '#38A169', '#9F7AEA', '#ED8936'
        ]
    
    def create_trends_chart(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """
        Create a trends/time series chart.
        
        Args:
            data: DataFrame with time series data
            title: Chart title
            filename: Base filename (without extension)
            
        Returns:
            Path to saved figure or None if failed
        """
        try:
            if data.empty or 'Year' not in data.columns or 'Value' not in data.columns:
                logger.warning("Insufficient data for trends chart")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Check if we have multiple series
            if 'Item' in data.columns and len(data['Item'].unique()) > 1:
                # Multiple items
                for i, item in enumerate(data['Item'].unique()):
                    item_data = data[data['Item'] == item]
                    ax.plot(
                        item_data['Year'], 
                        item_data['Value'], 
                        marker='o', 
                        linewidth=2.5,
                        markersize=4,
                        color=self.colors[i % len(self.colors)],
                        label=item
                    )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            elif 'Element' in data.columns and len(data['Element'].unique()) > 1:
                # Multiple elements
                for i, element in enumerate(data['Element'].unique()):
                    element_data = data[data['Element'] == element]
                    ax.plot(
                        element_data['Year'], 
                        element_data['Value'], 
                        marker='o', 
                        linewidth=2.5,
                        markersize=4,
                        color=self.colors[i % len(self.colors)],
                        label=element
                    )
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            else:
                # Single series
                ax.plot(
                    data['Year'], 
                    data['Value'], 
                    marker='o', 
                    linewidth=3,
                    markersize=5,
                    color=self.colors[0]
                )
            
            # Formatting
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis for large numbers
            ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_large_numbers))
            
            # Set year ticks
            years = sorted(data['Year'].unique())
            if len(years) > 10:
                # Show every nth year to avoid crowding
                step = max(1, len(years) // 8)
                ax.set_xticks(years[::step])
            
            plt.tight_layout()
            
            # Save figure
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Trends chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating trends chart: {str(e)}")
            plt.close()
            return None
    
    def create_rankings_chart(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """
        Create a rankings/bar chart.
        
        Args:
            data: DataFrame with rankings data
            title: Chart title
            filename: Base filename (without extension)
            
        Returns:
            Path to saved figure or None if failed
        """
        try:
            if data.empty or 'Area' not in data.columns or 'Value' not in data.columns:
                logger.warning("Insufficient data for rankings chart")
                return None
            
            # Take top 15 for readability
            plot_data = data.head(15).copy()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(
                range(len(plot_data)), 
                plot_data['Value'],
                color=self.colors[0],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Formatting
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Value', fontsize=11)
            ax.set_ylabel('Countries/Regions', fontsize=11)
            
            # Set y-axis labels
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['Area'], fontsize=9)
            
            # Invert y-axis to show highest values at top
            ax.invert_yaxis()
            
            # Format x-axis for large numbers
            ax.xaxis.set_major_formatter(plt.FuncFormatter(self._format_large_numbers))
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width + width * 0.01, 
                        bar.get_y() + bar.get_height()/2,
                        self._format_large_numbers(width, None),
                        ha='left', 
                        va='center',
                        fontsize=8,
                        color='#333333'
                    )
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save figure
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Rankings chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating rankings chart: {str(e)}")
            plt.close()
            return None
    
    def create_composition_chart(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """
        Create a composition chart (pie or stacked bar).
        
        Args:
            data: DataFrame with composition data
            title: Chart title
            filename: Base filename (without extension)
            
        Returns:
            Path to saved figure or None if failed
        """
        try:
            if data.empty or 'Value' not in data.columns:
                logger.warning("Insufficient data for composition chart")
                return None
            
            # Determine the category column
            category_col = None
            if 'Item' in data.columns:
                category_col = 'Item'
            elif 'Element' in data.columns:
                category_col = 'Element'
            else:
                logger.warning("No suitable category column for composition chart")
                return None
            
            # Prepare data - take top 8 categories and group rest as "Others"
            plot_data = data.copy()
            if len(plot_data) > 8:
                top_data = plot_data.head(7)
                others_value = plot_data.tail(len(plot_data) - 7)['Value'].sum()
                others_row = {category_col: 'Others', 'Value': others_value}
                plot_data = pd.concat([top_data, pd.DataFrame([others_row])], ignore_index=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            wedges, texts, autotexts = ax1.pie(
                plot_data['Value'],
                labels=plot_data[category_col],
                autopct='%1.1f%%',
                colors=self.colors[:len(plot_data)],
                startangle=90,
                textprops={'fontsize': 9}
            )
            
            ax1.set_title(f"{title} - Distribution", fontsize=12, fontweight='bold', pad=20)
            
            # Bar chart for absolute values
            bars = ax2.bar(
                range(len(plot_data)),
                plot_data['Value'],
                color=self.colors[:len(plot_data)],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            ax2.set_title(f"{title} - Absolute Values", fontsize=12, fontweight='bold', pad=20)
            ax2.set_xlabel(category_col, fontsize=11)
            ax2.set_ylabel('Value', fontsize=11)
            
            # Set x-axis labels with rotation
            ax2.set_xticks(range(len(plot_data)))
            ax2.set_xticklabels(plot_data[category_col], rotation=45, ha='right', fontsize=9)
            
            # Format y-axis for large numbers
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(self._format_large_numbers))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width()/2,
                        height + height * 0.01,
                        self._format_large_numbers(height, None),
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color='#333333'
                    )
            
            # Add grid to bar chart
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save figure
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Composition chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating composition chart: {str(e)}")
            plt.close()
            return None
    
    def create_regional_comparison(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """
        Create a regional comparison chart.
        
        Args:
            data: DataFrame with regional data
            title: Chart title
            filename: Base filename (without extension)
            
        Returns:
            Path to saved figure or None if failed
        """
        try:
            if data.empty or 'Area' not in data.columns or 'Value' not in data.columns:
                return None
            
            # Check if we have time dimension
            if 'Year' in data.columns:
                return self._create_multi_series_regional_chart(data, title, filename)
            else:
                return self._create_simple_regional_chart(data, title, filename)
                
        except Exception as e:
            logger.error(f"Error creating regional comparison: {str(e)}")
            plt.close()
            return None
    
    def _create_multi_series_regional_chart(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """Create regional chart with time series."""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top regions by total value
        region_totals = data.groupby('Area')['Value'].sum().sort_values(ascending=False)
        top_regions = region_totals.head(6).index.tolist()
        
        # Plot time series for top regions
        for i, region in enumerate(top_regions):
            region_data = data[data['Area'] == region].groupby('Year')['Value'].sum()
            ax.plot(
                region_data.index,
                region_data.values,
                marker='o',
                linewidth=2.5,
                markersize=4,
                color=self.colors[i % len(self.colors)],
                label=region
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_large_numbers))
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)
    
    def _create_simple_regional_chart(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """Create simple regional bar chart."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by region and sum
        regional_data = data.groupby('Area')['Value'].sum().sort_values(ascending=False)
        
        # Take top 10
        plot_data = regional_data.head(10)
        
        bars = ax.bar(
            range(len(plot_data)),
            plot_data.values,
            color=self.colors[0],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Regions', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        
        # Set x-axis labels
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=9)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_large_numbers))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"{filename}.png"
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)
    
    def _format_large_numbers(self, x, pos):
        """Format large numbers for axis labels."""
        if abs(x) >= 1e9:
            return f'{x/1e9:.1f}B'
        elif abs(x) >= 1e6:
            return f'{x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    def create_year_over_year_comparison(self, data: pd.DataFrame, title: str, filename: str) -> Optional[str]:
        """
        Create year-over-year comparison chart.
        
        Args:
            data: DataFrame with year-over-year data
            title: Chart title
            filename: Base filename (without extension)
            
        Returns:
            Path to saved figure or None if failed
        """
        try:
            if data.empty or 'Year' not in data.columns or 'Value' not in data.columns:
                return None
            
            # Get last two years
            years = sorted(data['Year'].unique())
            if len(years) < 2:
                return None
            
            current_year = years[-1]
            previous_year = years[-2]
            
            # Calculate changes by area
            current_data = data[data['Year'] == current_year].groupby('Area')['Value'].sum()
            previous_data = data[data['Year'] == previous_year].groupby('Area')['Value'].sum()
            
            # Calculate percent changes
            changes = {}
            for area in current_data.index:
                if area in previous_data.index and previous_data[area] != 0:
                    pct_change = ((current_data[area] - previous_data[area]) / previous_data[area]) * 100
                    changes[area] = {
                        'current': current_data[area],
                        'previous': previous_data[area],
                        'change_pct': pct_change
                    }
            
            if not changes:
                return None
            
            # Create DataFrame for plotting
            changes_df = pd.DataFrame(changes).T
            changes_df = changes_df.sort_values('change_pct', ascending=True)
            
            # Take top and bottom performers
            n_show = min(10, len(changes_df))
            plot_data = pd.concat([
                changes_df.head(n_show//2),  # Biggest decreases
                changes_df.tail(n_show//2)   # Biggest increases
            ])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color bars based on positive/negative change
            colors = ['#C73E1D' if x < 0 else '#38A169' for x in plot_data['change_pct']]
            
            bars = ax.barh(
                range(len(plot_data)),
                plot_data['change_pct'],
                color=colors,
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Formatting
            ax.set_title(f"{title}\nYear-over-Year Change ({previous_year} to {current_year})", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Percent Change (%)', fontsize=11)
            ax.set_ylabel('Countries/Regions', fontsize=11)
            
            # Set y-axis labels
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data.index, fontsize=9)
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width + (1 if width >= 0 else -1),
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left' if width >= 0 else 'right',
                    va='center',
                    fontsize=8,
                    color='#333333'
                )
            
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Year-over-year comparison saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating year-over-year comparison: {str(e)}")
            plt.close()
            return None