"""
Brief Generator Service for creating FAO-style analytical briefs.

This service orchestrates the generation of comprehensive analytical briefs
following the FAO FAOSTAT Analytical Brief format with AI-powered insights.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class BriefGeneratorService:
    """Service for generating FAO-style analytical briefs."""
    
    def __init__(self, openai_service, faostat_service, geography_service, viz_generator):
        """
        Initialize brief generator service.
        
        Args:
            openai_service: OpenAI service for AI-powered insights
            faostat_service: FAOSTAT service for data access
            geography_service: Geography service for area classification
            viz_generator: Visualization generator for charts
        """
        self.openai_service = openai_service
        self.faostat_service = faostat_service
        self.geography_service = geography_service
        self.viz_generator = viz_generator
    
    def generate_brief(self, dataset_code: str, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete analytical brief.
        
        Args:
            dataset_code: FAOSTAT dataset code
            dataset_name: Human-readable dataset name
            config: Analysis configuration from the UI
            
        Returns:
            Dictionary containing the generated brief and all components
        """
        logger.info(f"Generating brief for dataset {dataset_code}")
        
        try:
            # Extract configuration
            df = config['dataset_df']
            year_range = config.get('year_range')
            selected_items = config.get('selected_items', [])
            selected_areas = config.get('selected_areas', [])
            selected_elements = config.get('selected_elements', [])
            geo_scope = config.get('geo_scope', 'Countries')
            
            # Filter the dataset
            filtered_df = self._filter_dataset(df, config)
            
            # Generate summary statistics
            summary_stats = self._generate_summary_stats(filtered_df, config)
            
            # Generate visualizations
            figures = self._generate_visualizations(filtered_df, config)
            
            # Generate AI insights
            insights = self._generate_insights(filtered_df, summary_stats, config)
            
            # Package results
            brief_results = {
                'dataset_code': dataset_code,
                'dataset_name': dataset_name,
                'config': config,
                'filtered_data': filtered_df,
                'summary_stats': summary_stats,
                'figures': figures,
                'insights': insights,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info("Brief generation completed successfully")
            return brief_results
            
        except Exception as e:
            logger.error(f"Error generating brief: {str(e)}")
            raise
    
    def _filter_dataset(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Filter dataset based on configuration."""
        
        filtered_df = df.copy()
        
        # Apply year filter
        year_range = config.get('year_range')
        if year_range and 'Year' in filtered_df.columns:
            start_year, end_year = year_range
            filtered_df = filtered_df[
                (filtered_df['Year'] >= start_year) & 
                (filtered_df['Year'] <= end_year)
            ]
        
        # Apply area filter
        selected_areas = config.get('selected_areas')
        if selected_areas and 'Area' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Area'].isin(selected_areas)]
        
        # Apply item filter
        selected_items = config.get('selected_items')
        if selected_items and 'Item' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Item'].isin(selected_items)]
        
        # Apply element filter
        selected_elements = config.get('selected_elements')
        if selected_elements and 'Element' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Element'].isin(selected_elements)]
        
        return filtered_df
    
    def _generate_summary_stats(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the filtered dataset."""
        
        stats = {
            'total_records': len(df),
            'year_range': None,
            'areas_covered': [],
            'items_covered': [],
            'elements_covered': []
        }
        
        if df.empty:
            return stats
        
        # Year range
        if 'Year' in df.columns:
            stats['year_range'] = (int(df['Year'].min()), int(df['Year'].max()))
        
        # Geographic coverage
        if 'Area' in df.columns:
            stats['areas_covered'] = sorted(df['Area'].unique().tolist())
            stats['num_areas'] = len(stats['areas_covered'])
        
        # Items covered
        if 'Item' in df.columns:
            stats['items_covered'] = sorted(df['Item'].unique().tolist())
            stats['num_items'] = len(stats['items_covered'])
        
        # Elements covered
        if 'Element' in df.columns:
            stats['elements_covered'] = sorted(df['Element'].unique().tolist())
            stats['num_elements'] = len(stats['elements_covered'])
        
        # Value statistics
        if 'Value' in df.columns:
            stats['total_value'] = df['Value'].sum()
            stats['avg_value'] = df['Value'].mean()
            stats['value_by_year'] = self._get_yearly_trends(df)
            stats['value_by_area'] = self._get_area_rankings(df)
            stats['latest_year_data'] = self._get_latest_year_analysis(df)
            stats['year_over_year_change'] = self._get_year_over_year_changes(df)
        
        return stats
    
    def _get_yearly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate yearly trends."""
        if 'Year' not in df.columns or 'Value' not in df.columns:
            return {}
        
        yearly_totals = df.groupby('Year')['Value'].sum()
        
        return {
            'by_year': yearly_totals.to_dict(),
            'trend_direction': 'increasing' if yearly_totals.iloc[-1] > yearly_totals.iloc[0] else 'decreasing',
            'percent_change_total': ((yearly_totals.iloc[-1] - yearly_totals.iloc[0]) / yearly_totals.iloc[0] * 100) if yearly_totals.iloc[0] != 0 else 0
        }
    
    def _get_area_rankings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate area rankings."""
        if 'Area' not in df.columns or 'Value' not in df.columns:
            return {}
        
        area_totals = df.groupby('Area')['Value'].sum().sort_values(ascending=False)
        
        return {
            'top_10': area_totals.head(10).to_dict(),
            'bottom_10': area_totals.tail(10).to_dict(),
            'total_areas': len(area_totals)
        }
    
    def _get_latest_year_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze latest year data."""
        if 'Year' not in df.columns or 'Value' not in df.columns:
            return {}
        
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year]
        
        result = {
            'latest_year': latest_year,
            'total_value': latest_data['Value'].sum(),
            'num_records': len(latest_data)
        }
        
        # Top areas in latest year
        if 'Area' in latest_data.columns:
            area_totals = latest_data.groupby('Area')['Value'].sum().sort_values(ascending=False)
            result['top_areas'] = area_totals.head(10).to_dict()
        
        # Top items in latest year
        if 'Item' in latest_data.columns:
            item_totals = latest_data.groupby('Item')['Value'].sum().sort_values(ascending=False)
            result['top_items'] = item_totals.head(10).to_dict()
        
        return result
    
    def _get_year_over_year_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate year-over-year changes."""
        if 'Year' not in df.columns or 'Value' not in df.columns:
            return {}
        
        years = sorted(df['Year'].unique())
        if len(years) < 2:
            return {}
        
        latest_year = years[-1]
        previous_year = years[-2]
        
        latest_total = df[df['Year'] == latest_year]['Value'].sum()
        previous_total = df[df['Year'] == previous_year]['Value'].sum()
        
        percent_change = ((latest_total - previous_total) / previous_total * 100) if previous_total != 0 else 0
        
        result = {
            'latest_year': latest_year,
            'previous_year': previous_year,
            'latest_total': latest_total,
            'previous_total': previous_total,
            'absolute_change': latest_total - previous_total,
            'percent_change': percent_change,
            'direction': 'increase' if percent_change > 0 else 'decrease' if percent_change < 0 else 'no change'
        }
        
        # Calculate changes by area
        if 'Area' in df.columns:
            area_changes = {}
            for area in df['Area'].unique():
                area_data = df[df['Area'] == area]
                latest_area = area_data[area_data['Year'] == latest_year]['Value'].sum()
                previous_area = area_data[area_data['Year'] == previous_year]['Value'].sum()
                
                if previous_area != 0:
                    area_change = ((latest_area - previous_area) / previous_area * 100)
                    area_changes[area] = {
                        'percent_change': area_change,
                        'absolute_change': latest_area - previous_area,
                        'latest_value': latest_area,
                        'previous_value': previous_area
                    }
            
            # Get top positive and negative changes
            sorted_changes = sorted(area_changes.items(), key=lambda x: x[1]['percent_change'], reverse=True)
            result['top_increases'] = dict(sorted_changes[:5])
            result['top_decreases'] = dict(sorted_changes[-5:])
        
        return result
    
    def _generate_visualizations(self, df: pd.DataFrame, config: Dict[str, Any]) -> List[Any]:
        """Generate visualizations for the brief."""
        
        figures = []
        
        if df.empty:
            return figures
        
        try:
            # Figure 1: Global trends by item and element
            if 'Year' in df.columns and 'Value' in df.columns:
                fig1_data = self._prepare_trends_data(df, config)
                if not fig1_data.empty:
                    fig1_path = self.viz_generator.create_trends_chart(
                        fig1_data,
                        title="Global Trends by Item and Element",
                        filename="fig1_global_trends"
                    )
                    if fig1_path:
                        figures.append({
                            'type': 'trends',
                            'title': 'Global Trends by Item and Element',
                            'filename': 'fig1_global_trends.png',
                            'filepath': fig1_path,
                            'description': 'Time series showing global trends for selected items and elements',
                            'data_sheet': 'fig1_data'
                        })
            
            # Figure 2: Regional/Country rankings  
            if 'Area' in df.columns and 'Value' in df.columns:
                fig2_data = self._prepare_rankings_data(df, config)
                if not fig2_data.empty:
                    fig2_path = self.viz_generator.create_rankings_chart(
                        fig2_data,
                        title="Rankings by Geographic Area",
                        filename="fig2_rankings"
                    )
                    if fig2_path:
                        figures.append({
                            'type': 'rankings',
                            'title': 'Rankings by Geographic Area',
                            'filename': 'fig2_rankings.png',
                            'filepath': fig2_path,
                            'description': 'Bar chart showing top areas by total value',
                            'data_sheet': 'fig2_data'
                        })
            
            # Figure 3: Item/Element breakdown
            if 'Item' in df.columns and len(df['Item'].unique()) > 1:
                fig3_data = self._prepare_composition_data(df, config)
                if not fig3_data.empty:
                    fig3_path = self.viz_generator.create_composition_chart(
                        fig3_data,
                        title="Composition by Item and Element",
                        filename="fig3_composition"
                    )
                    if fig3_path:
                        figures.append({
                            'type': 'composition',
                            'title': 'Composition by Item and Element',
                            'filename': 'fig3_composition.png',
                            'filepath': fig3_path,
                            'description': 'Breakdown showing relative importance of different items and elements',
                            'data_sheet': 'fig3_data'
                        })
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        return figures
    
    def _prepare_trends_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for trends visualization."""
        if 'Year' not in df.columns or 'Value' not in df.columns:
            return pd.DataFrame()
        
        # Group by year and sum values
        trends_data = df.groupby('Year')['Value'].sum().reset_index()
        
        # If we have items/elements, create separate series
        if 'Item' in df.columns and len(df['Item'].unique()) > 1:
            trends_data = df.groupby(['Year', 'Item'])['Value'].sum().reset_index()
        elif 'Element' in df.columns and len(df['Element'].unique()) > 1:
            trends_data = df.groupby(['Year', 'Element'])['Value'].sum().reset_index()
        
        return trends_data
    
    def _prepare_rankings_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for rankings visualization."""
        if 'Area' not in df.columns or 'Value' not in df.columns:
            return pd.DataFrame()
        
        # Get latest year data for rankings
        if 'Year' in df.columns:
            latest_year = df['Year'].max()
            rankings_data = df[df['Year'] == latest_year]
        else:
            rankings_data = df
        
        # Group by area and sum values
        area_totals = rankings_data.groupby('Area')['Value'].sum().sort_values(ascending=False)
        
        # Take top 15 for readability
        top_areas = area_totals.head(15).reset_index()
        
        return top_areas
    
    def _prepare_composition_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for composition visualization."""
        if 'Value' not in df.columns:
            return pd.DataFrame()
        
        # Use items if available, otherwise elements
        if 'Item' in df.columns and len(df['Item'].unique()) > 1:
            composition_data = df.groupby('Item')['Value'].sum().sort_values(ascending=False)
        elif 'Element' in df.columns and len(df['Element'].unique()) > 1:
            composition_data = df.groupby('Element')['Value'].sum().sort_values(ascending=False)
        else:
            return pd.DataFrame()
        
        return composition_data.reset_index()
    
    def _generate_insights(self, df: pd.DataFrame, summary_stats: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights for the brief."""
        
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(df, summary_stats, config)
            
            # Generate insights using OpenAI
            insights = {}
            
            # Generate highlights
            insights['highlights'] = self._generate_highlights(context)
            
            # Generate global trends analysis
            insights['global_trends'] = self._generate_global_trends_analysis(context)
            
            # Generate regional/country analysis
            insights['regional_analysis'] = self._generate_regional_analysis(context)
            
            # Generate country-specific insights
            insights['country_insights'] = self._generate_country_insights(context)
            
            # Generate explanatory notes
            insights['explanatory_notes'] = self._generate_explanatory_notes(context)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return self._get_fallback_insights(summary_stats)
    
    def _prepare_analysis_context(self, df: pd.DataFrame, summary_stats: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for AI analysis."""
        
        context = {
            'dataset_info': {
                'name': config.get('dataset_name', 'Unknown Dataset'),
                'code': config.get('dataset_code', 'Unknown'),
                'total_records': len(df),
                'time_period': summary_stats.get('year_range'),
                'geographic_scope': config.get('geo_scope', 'Unknown'),
                'selected_items': config.get('selected_items', []),
                'selected_elements': config.get('selected_elements', []),
                'selected_areas': config.get('selected_areas', [])
            },
            'summary_stats': summary_stats,
            'key_findings': {
                'latest_year_total': summary_stats.get('latest_year_data', {}).get('total_value', 0),
                'year_over_year_change': summary_stats.get('year_over_year_change', {}),
                'top_areas': list(summary_stats.get('value_by_area', {}).get('top_10', {}).keys())[:5],
                'trend_direction': summary_stats.get('value_by_year', {}).get('trend_direction', 'unknown')
            }
        }
        
        return context
    
    def _generate_highlights(self, context: Dict[str, Any]) -> str:
        """Generate key highlights section."""
        
        prompt = f"""
        Generate key highlights for a FAOSTAT analytical brief based on the following data:
        
        Dataset: {context['dataset_info']['name']}
        Time Period: {context['dataset_info']['time_period']}
        Geographic Scope: {context['dataset_info']['geographic_scope']}
        Items: {', '.join(context['dataset_info']['selected_items'][:3])}
        Elements: {', '.join(context['dataset_info']['selected_elements'][:3])}
        
        Key Statistics:
        - Latest year total: {context['key_findings']['latest_year_total']:,.0f}
        - Year-over-year change: {context['key_findings']['year_over_year_change'].get('percent_change', 0):.1f}%
        - Trend direction: {context['key_findings']['trend_direction']}
        - Top areas: {', '.join(context['key_findings']['top_areas'])}
        
        Generate 3-5 bullet points highlighting the most important findings in the style of FAO analytical briefs.
        Focus on quantitative insights and trends. Use professional, factual language.
        """
        
        try:
            response = self.openai_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating highlights: {str(e)}")
            return self._get_fallback_highlights(context)
    
    def _generate_global_trends_analysis(self, context: Dict[str, Any]) -> str:
        """Generate global trends analysis section."""
        
        yoy_change = context['key_findings']['year_over_year_change']
        
        prompt = f"""
        Generate a global trends analysis for a FAOSTAT analytical brief:
        
        Dataset: {context['dataset_info']['name']}
        Time Period: {context['dataset_info']['time_period']}
        
        Year-over-Year Changes:
        - Latest year: {yoy_change.get('latest_year', 'N/A')}
        - Previous year: {yoy_change.get('previous_year', 'N/A')}
        - Percent change: {yoy_change.get('percent_change', 0):.1f}%
        - Direction: {yoy_change.get('direction', 'unknown')}
        
        Write a detailed analysis (150-200 words) explaining:
        1. Overall global trends for the selected items and elements
        2. Changes between the latest and previous year
        3. Possible factors explaining these trends
        4. Context about longer-term patterns
        
        Use the style of FAO analytical briefs - professional, factual, with specific numbers and percentages.
        """
        
        try:
            response = self.openai_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating global trends: {str(e)}")
            return f"Global analysis shows a {yoy_change.get('percent_change', 0):.1f}% change between {yoy_change.get('previous_year', 'N/A')} and {yoy_change.get('latest_year', 'N/A')}."
    
    def _generate_regional_analysis(self, context: Dict[str, Any]) -> str:
        """Generate regional analysis section."""
        
        top_areas = context['key_findings']['top_areas']
        geo_scope = context['dataset_info']['geographic_scope']
        
        prompt = f"""
        Generate a regional analysis for a FAOSTAT analytical brief:
        
        Geographic Scope: {geo_scope}
        Top Areas: {', '.join(top_areas)}
        Dataset: {context['dataset_info']['name']}
        
        Write a detailed analysis (150-200 words) covering:
        1. Performance of different regions/countries
        2. Leading areas and their contributions
        3. Regional patterns and differences
        4. Notable changes or trends by geography
        
        Use FAO analytical brief style with specific data points and professional language.
        """
        
        try:
            response = self.openai_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating regional analysis: {str(e)}")
            return f"The analysis covers {geo_scope.lower()} level data, with {', '.join(top_areas[:3])} among the leading areas."
    
    def _generate_country_insights(self, context: Dict[str, Any]) -> str:
        """Generate country-specific insights."""
        
        yoy_data = context['key_findings']['year_over_year_change']
        
        prompt = f"""
        Generate country-level insights for a FAOSTAT analytical brief:
        
        Dataset: {context['dataset_info']['name']}
        
        Top Performers: {', '.join(context['key_findings']['top_areas'])}
        
        Year-over-Year Changes by Area:
        Top Increases: {list(yoy_data.get('top_increases', {}).keys())[:3]}
        Top Decreases: {list(yoy_data.get('top_decreases', {}).keys())[:3]}
        
        Write an analysis (100-150 words) highlighting:
        1. Countries with notable performance
        2. Significant increases or decreases
        3. Interesting patterns or outliers
        4. Country-specific factors that may explain changes
        
        Use FAO style with specific country examples and data.
        """
        
        try:
            response = self.openai_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating country insights: {str(e)}")
            return f"Country analysis shows varied performance across {context['dataset_info']['geographic_scope'].lower()}."
    
    def _generate_explanatory_notes(self, context: Dict[str, Any]) -> str:
        """Generate explanatory notes section."""
        
        prompt = f"""
        Generate explanatory notes for a FAOSTAT analytical brief:
        
        Dataset: {context['dataset_info']['name']} ({context['dataset_info']['code']})
        
        Write brief explanatory notes (80-100 words) covering:
        1. Data source and methodology
        2. Any limitations or caveats
        3. Definition of key terms if needed
        4. Update frequency or data availability notes
        
        Use professional, concise language typical of FAO documentation.
        """
        
        try:
            response = self.openai_service.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating explanatory notes: {str(e)}")
            return f"Data sourced from FAOSTAT database ({context['dataset_info']['code']}). Statistics compiled from official country submissions and FAO estimates."
    
    def _get_fallback_insights(self, summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback insights if AI generation fails."""
        
        return {
            'highlights': f"Analysis covers {summary_stats.get('num_areas', 0)} areas and {summary_stats.get('total_records', 0)} data points.",
            'global_trends': "Global trends analysis based on available data points.",
            'regional_analysis': f"Regional analysis covers {summary_stats.get('num_areas', 0)} geographic areas.",
            'country_insights': "Country-level analysis shows varied performance across different areas.",
            'explanatory_notes': f"Data sourced from FAOSTAT. Analysis period: {summary_stats.get('year_range', 'Not specified')}."
        }
    
    def _get_fallback_highlights(self, context: Dict[str, Any]) -> str:
        """Generate fallback highlights if AI fails."""
        
        yoy_change = context['key_findings']['year_over_year_change']
        latest_total = context['key_findings']['latest_year_total']
        
        highlights = []
        
        if latest_total:
            highlights.append(f"→ Total value in latest year: {latest_total:,.0f}")
        
        if yoy_change.get('percent_change'):
            change_pct = yoy_change['percent_change']
            direction = "increased" if change_pct > 0 else "decreased"
            highlights.append(f"→ Year-over-year change: {direction} by {abs(change_pct):.1f}%")
        
        if context['key_findings']['top_areas']:
            top_3 = ', '.join(context['key_findings']['top_areas'][:3])
            highlights.append(f"→ Leading areas: {top_3}")
        
        if context['dataset_info']['selected_items']:
            items = ', '.join(context['dataset_info']['selected_items'][:2])
            highlights.append(f"→ Analysis covers: {items}")
        
        return '\n'.join(highlights)