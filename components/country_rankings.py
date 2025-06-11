"""
Country rankings analysis component.

This module provides components for analyzing and displaying country rankings
based on various agricultural indicators and metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from utils.data_processing import identify_top_bottom_performers
from utils.prompt_templates import get_rankings_analysis_prompt

logger = logging.getLogger(__name__)

def render_country_rankings_analysis(
    dataset_df: pd.DataFrame,
    value_column: str = 'Value',
    year_filter: Optional[int] = None,
    top_n: int = 10,
    bottom_n: int = 10
):
    """
    Render comprehensive country rankings analysis.
    
    Args:
        dataset_df: Dataset containing country data
        value_column: Column to rank countries by
        year_filter: Specific year to analyze (None for all years)
        top_n: Number of top performers to show
        bottom_n: Number of bottom performers to show
    """
    
    if 'Area' not in dataset_df.columns:
        st.error("❌ This dataset doesn't contain country/area information")
        return
    
    st.markdown("## 🏆 Country Rankings Analysis")
    
    # Configuration
    render_rankings_configuration(dataset_df, year_filter, top_n, bottom_n)
    
    # Get current configuration
    config = get_rankings_configuration()
    
    # Perform analysis
    rankings_data = perform_rankings_analysis(dataset_df, config)
    
    if not rankings_data:
        st.error("❌ Could not perform rankings analysis")
        return
    
    # Display results
    render_rankings_results(rankings_data, config)
    
    # AI insights if available
    render_rankings_insights(rankings_data, config)

def render_rankings_configuration(
    dataset_df: pd.DataFrame,
    year_filter: Optional[int],
    top_n: int,
    bottom_n: int
):
    """Render configuration options for rankings analysis."""
    
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year selection
        if 'Year' in dataset_df.columns:
            available_years = sorted(dataset_df['Year'].unique(), reverse=True)
            
            year_option = st.selectbox(
                "Analysis Year",
                options=["Latest available"] + available_years,
                key="rankings_year"
            )
            
            if year_option != "Latest available":
                st.session_state.rankings_year_filter = int(year_option)
            else:
                st.session_state.rankings_year_filter = available_years[0]
        else:
            st.info("No year information available")
            st.session_state.rankings_year_filter = None
    
    with col2:
        # Number of countries to show
        st.session_state.rankings_top_n = st.slider(
            "Top performers",
            min_value=5,
            max_value=20,
            value=top_n,
            key="rankings_top_slider"
        )
        
        st.session_state.rankings_bottom_n = st.slider(
            "Bottom performers", 
            min_value=5,
            max_value=20,
            value=bottom_n,
            key="rankings_bottom_slider"
        )
    
    with col3:
        # Metric selection
        value_columns = ['Value']
        if 'Production' in dataset_df.columns:
            value_columns.append('Production')
        if 'Yield' in dataset_df.columns:
            value_columns.append('Yield')
        
        st.session_state.rankings_metric = st.selectbox(
            "Ranking Metric",
            options=value_columns,
            key="rankings_metric_select"
        )
        
        # Aggregation method
        st.session_state.rankings_aggregation = st.selectbox(
            "Aggregation",
            options=["sum", "mean", "median", "max"],
            index=0,
            key="rankings_agg_select"
        )

def get_rankings_configuration() -> Dict[str, Any]:
    """Get current rankings configuration from session state."""
    
    return {
        'year_filter': st.session_state.get('rankings_year_filter'),
        'top_n': st.session_state.get('rankings_top_n', 10),
        'bottom_n': st.session_state.get('rankings_bottom_n', 10),
        'metric': st.session_state.get('rankings_metric', 'Value'),
        'aggregation': st.session_state.get('rankings_aggregation', 'sum')
    }

def perform_rankings_analysis(
    dataset_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Perform the rankings analysis."""
    
    try:
        # Filter data for the specified year if provided
        analysis_df = dataset_df.copy()
        
        if config['year_filter'] and 'Year' in analysis_df.columns:
            analysis_df = analysis_df[analysis_df['Year'] == config['year_filter']]
        
        if analysis_df.empty:
            st.error("No data available for the selected year")
            return None
        
        # Aggregate data by country
        agg_method = config['aggregation']
        metric_column = config['metric']
        
        if metric_column not in analysis_df.columns:
            st.error(f"Column '{metric_column}' not found in dataset")
            return None
        
        # Group by country and aggregate
        if agg_method == 'sum':
            country_data = analysis_df.groupby('Area')[metric_column].sum().reset_index()
        elif agg_method == 'mean':
            country_data = analysis_df.groupby('Area')[metric_column].mean().reset_index()
        elif agg_method == 'median':
            country_data = analysis_df.groupby('Area')[metric_column].median().reset_index()
        elif agg_method == 'max':
            country_data = analysis_df.groupby('Area')[metric_column].max().reset_index()
        
        # Sort and get top/bottom performers
        country_data = country_data.sort_values(metric_column, ascending=False)
        
        top_performers = country_data.head(config['top_n'])
        bottom_performers = country_data.tail(config['bottom_n']).sort_values(metric_column, ascending=True)
        
        # Calculate statistics
        stats = calculate_rankings_statistics(country_data, metric_column)
        
        return {
            'top_performers': top_performers,
            'bottom_performers': bottom_performers,
            'all_countries': country_data,
            'statistics': stats,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Error in rankings analysis: {str(e)}")
        st.error(f"Error performing analysis: {str(e)}")
        return None

def calculate_rankings_statistics(
    country_data: pd.DataFrame,
    metric_column: str
) -> Dict[str, float]:
    """Calculate summary statistics for the rankings."""
    
    stats = {
        'total_countries': len(country_data),
        'global_mean': country_data[metric_column].mean(),
        'global_median': country_data[metric_column].median(),
        'global_std': country_data[metric_column].std(),
        'top10_mean': country_data.head(10)[metric_column].mean(),
        'bottom10_mean': country_data.tail(10)[metric_column].mean(),
    }
    
    # Calculate ratio
    if stats['bottom10_mean'] > 0:
        stats['top10_to_bottom10_ratio'] = stats['top10_mean'] / stats['bottom10_mean']
    else:
        stats['top10_to_bottom10_ratio'] = float('inf')
    
    return stats

def render_rankings_results(rankings_data: Dict[str, Any], config: Dict[str, Any]):
    """Render the rankings analysis results."""
    
    st.markdown("### 📊 Rankings Results")
    
    # Summary statistics
    render_rankings_summary(rankings_data['statistics'], config)
    
    # Top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        render_top_performers(rankings_data['top_performers'], config)
    
    with col2:
        render_bottom_performers(rankings_data['bottom_performers'], config)
    
    # Full rankings table
    render_full_rankings(rankings_data['all_countries'], config)
    
    # Visualizations
    render_rankings_visualizations(rankings_data, config)

def render_rankings_summary(statistics: Dict[str, float], config: Dict[str, Any]):
    """Render summary statistics."""
    
    st.markdown("#### 📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Countries",
            f"{statistics['total_countries']:,}"
        )
    
    with col2:
        st.metric(
            "Global Average",
            f"{statistics['global_mean']:,.1f}"
        )
    
    with col3:
        st.metric(
            "Top 10 Average",
            f"{statistics['top10_mean']:,.1f}",
            delta=f"{statistics['top10_mean'] - statistics['global_mean']:,.1f}"
        )
    
    with col4:
        st.metric(
            "Top/Bottom Ratio",
            f"{statistics['top10_to_bottom10_ratio']:.1f}x"
        )

def render_top_performers(top_performers: pd.DataFrame, config: Dict[str, Any]):
    """Render top performing countries."""
    
    st.markdown(f"#### 🥇 Top {len(top_performers)} Performers")
    
    # Add ranking column
    top_performers_display = top_performers.copy()
    top_performers_display['Rank'] = range(1, len(top_performers) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Area', config['metric']]
    top_performers_display = top_performers_display[cols]
    
    # Format values
    top_performers_display[config['metric']] = top_performers_display[config['metric']].round(1)
    
    st.dataframe(
        top_performers_display,
        hide_index=True,
        use_container_width=True
    )
    
    # Highlight top 3
    if len(top_performers) >= 3:
        st.markdown("**🏆 Podium:**")
        st.success(f"🥇 Gold: {top_performers.iloc[0]['Area']}")
        st.info(f"🥈 Silver: {top_performers.iloc[1]['Area']}")
        st.warning(f"🥉 Bronze: {top_performers.iloc[2]['Area']}")

def render_bottom_performers(bottom_performers: pd.DataFrame, config: Dict[str, Any]):
    """Render bottom performing countries."""
    
    st.markdown(f"#### 📉 Bottom {len(bottom_performers)} Performers")
    
    # Add ranking column (from bottom)
    bottom_performers_display = bottom_performers.copy()
    bottom_performers_display['Rank'] = range(len(bottom_performers), 0, -1)
    
    # Reorder columns
    cols = ['Rank', 'Area', config['metric']]
    bottom_performers_display = bottom_performers_display[cols]
    
    # Format values
    bottom_performers_display[config['metric']] = bottom_performers_display[config['metric']].round(1)
    
    st.dataframe(
        bottom_performers_display,
        hide_index=True,
        use_container_width=True
    )

def render_full_rankings(all_countries: pd.DataFrame, config: Dict[str, Any]):
    """Render full rankings table."""
    
    with st.expander(f"📋 Complete Rankings ({len(all_countries)} countries)"):
        
        # Add ranking
        all_countries_display = all_countries.copy()
        all_countries_display['Rank'] = range(1, len(all_countries) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Area', config['metric']]
        all_countries_display = all_countries_display[cols]
        
        # Format values
        all_countries_display[config['metric']] = all_countries_display[config['metric']].round(1)
        
        st.dataframe(
            all_countries_display,
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv_data = all_countries_display.to_csv(index=False)
        st.download_button(
            label="💾 Download Rankings CSV",
            data=csv_data,
            file_name=f"country_rankings_{config['metric']}_{config.get('year_filter', 'all_years')}.csv",
            mime="text/csv"
        )

def render_rankings_visualizations(rankings_data: Dict[str, Any], config: Dict[str, Any]):
    """Render visualizations for rankings data."""
    
    st.markdown("#### 📊 Rankings Visualizations")
    
    # Create visualizations
    fig_info = None
    
    if hasattr(st.session_state, 'viz_generator'):
        try:
            viz_generator = st.session_state.viz_generator
            
            # Create comparison visualization
            fig_info = viz_generator.create_top_bottom_comparison(
                top_entities=rankings_data['top_performers'],
                bottom_entities=rankings_data['bottom_performers'],
                entity_column='Area',
                value_column=config['metric'],
                title_prefix=f"Country Rankings by {config['metric']}"
            )
            
            if fig_info and fig_info.filepath:
                st.image(fig_info.filepath, caption=fig_info.title)
        
        except Exception as e:
            logger.error(f"Error creating rankings visualization: {str(e)}")
            st.error(f"Could not create visualization: {str(e)}")
    
    # Simple bar chart using Streamlit
    st.markdown("**Top 10 vs Bottom 10 Comparison**")
    
    # Combine top and bottom for comparison
    top_5 = rankings_data['top_performers'].head(5)
    bottom_5 = rankings_data['bottom_performers'].head(5)
    
    # Create chart data
    chart_data = pd.concat([
        top_5[['Area', config['metric']]],
        bottom_5[['Area', config['metric']]]
    ])
    
    chart_data = chart_data.set_index('Area')
    st.bar_chart(chart_data)

def render_rankings_insights(rankings_data: Dict[str, Any], config: Dict[str, Any]):
    """Render AI-powered insights for rankings analysis."""
    
    st.markdown("### 🤖 AI Analysis")
    
    openai_service = st.session_state.get('openai_service')
    
    if not openai_service:
        st.info("💡 Configure OpenAI API key to get AI-powered insights about these rankings")
        return
    
    if st.button("🧠 Generate Rankings Insights", type="primary"):
        with st.spinner("🤖 AI is analyzing country rankings..."):
            try:
                # Prepare data for AI analysis
                top_countries_list = []
                for _, row in rankings_data['top_performers'].iterrows():
                    top_countries_list.append({
                        'country': row['Area'],
                        'value': float(row[config['metric']]),
                        'rank': len(top_countries_list) + 1
                    })
                
                bottom_countries_list = []
                for _, row in rankings_data['bottom_performers'].iterrows():
                    bottom_countries_list.append({
                        'country': row['Area'],
                        'value': float(row[config['metric']]),
                        'rank': rankings_data['statistics']['total_countries'] - len(bottom_countries_list)
                    })
                
                # Generate insights using prompt template
                prompt = get_rankings_analysis_prompt(
                    top_countries=top_countries_list,
                    bottom_countries=bottom_countries_list,
                    stats=rankings_data['statistics']
                )
                
                # Get AI response
                messages = [
                    {"role": "system", "content": "You are an expert agricultural economist analyzing country performance data."},
                    {"role": "user", "content": prompt}
                ]
                
                insights = openai_service._call_openai_api(messages)
                
                # Display insights
                st.markdown("#### 🔍 AI Insights")
                st.markdown(insights)
                
                # Store insights
                st.session_state.rankings_insights = insights
                
            except Exception as e:
                logger.error(f"Error generating rankings insights: {str(e)}")
                st.error(f"Error generating insights: {str(e)}")

def render_rankings_comparison_over_time(
    dataset_df: pd.DataFrame,
    countries: List[str],
    metric: str = 'Value'
):
    """Render rankings comparison over time for selected countries."""
    
    if 'Year' not in dataset_df.columns:
        st.info("Time comparison not available - no year data")
        return
    
    st.markdown("#### 📈 Rankings Over Time")
    
    # Filter for selected countries
    time_df = dataset_df[dataset_df['Area'].isin(countries)].copy()
    
    if time_df.empty:
        st.warning("No data available for selected countries")
        return
    
    # Calculate rankings for each year
    yearly_rankings = []
    
    for year in sorted(time_df['Year'].unique()):
        year_data = time_df[time_df['Year'] == year]
        year_totals = year_data.groupby('Area')[metric].sum().reset_index()
        year_totals['Rank'] = year_totals[metric].rank(method='dense', ascending=False)
        year_totals['Year'] = year
        yearly_rankings.append(year_totals)
    
    rankings_df = pd.concat(yearly_rankings)
    
    # Create line chart of rankings over time
    pivot_rankings = rankings_df.pivot(index='Year', columns='Area', values='Rank')
    
    st.line_chart(pivot_rankings)
    
    # Show data table
    with st.expander("📊 Detailed Rankings by Year"):
        display_df = rankings_df.pivot(index='Year', columns='Area', values='Rank')
        st.dataframe(display_df, use_container_width=True)

def render_regional_rankings_analysis(
    dataset_df: pd.DataFrame,
    metric: str = 'Value'
):
    """Render rankings analysis grouped by regions."""
    
    if 'Area' not in dataset_df.columns:
        return
    
    st.markdown("#### 🌍 Regional Analysis")
    
    # Define regional groupings (simplified)
    regional_mapping = get_regional_mapping()
    
    # Add region column
    dataset_with_regions = dataset_df.copy()
    dataset_with_regions['Region'] = dataset_with_regions['Area'].map(regional_mapping)
    dataset_with_regions['Region'] = dataset_with_regions['Region'].fillna('Other')
    
    # Calculate regional totals
    regional_totals = dataset_with_regions.groupby('Region')[metric].sum().reset_index()
    regional_totals = regional_totals.sort_values(metric, ascending=False)
    
    # Display regional rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Regional Totals:**")
        st.dataframe(regional_totals, hide_index=True)
    
    with col2:
        st.markdown("**Regional Distribution:**")
        st.bar_chart(regional_totals.set_index('Region')[metric])

def get_regional_mapping() -> Dict[str, str]:
    """Get mapping of countries to regions."""
    
    # Simplified regional mapping - in practice, this could come from a more comprehensive source
    return {
        # Africa
        'Nigeria': 'Africa', 'Egypt': 'Africa', 'South Africa': 'Africa',
        'Kenya': 'Africa', 'Ethiopia': 'Africa', 'Morocco': 'Africa',
        
        # Asia
        'China': 'Asia', 'India': 'Asia', 'Indonesia': 'Asia',
        'Japan': 'Asia', 'Thailand': 'Asia', 'Viet Nam': 'Asia',
        'Bangladesh': 'Asia', 'Pakistan': 'Asia', 'Philippines': 'Asia',
        
        # Europe
        'Germany': 'Europe', 'France': 'Europe', 'United Kingdom': 'Europe',
        'Italy': 'Europe', 'Spain': 'Europe', 'Poland': 'Europe',
        'Russian Federation': 'Europe', 'Ukraine': 'Europe',
        
        # Americas
        'United States of America': 'Americas', 'Brazil': 'Americas',
        'Canada': 'Americas', 'Mexico': 'Americas', 'Argentina': 'Americas',
        'Colombia': 'Americas', 'Chile': 'Americas', 'Peru': 'Americas',
        
        # Oceania
        'Australia': 'Oceania', 'New Zealand': 'Oceania'
    }

def export_rankings_report(rankings_data: Dict[str, Any], config: Dict[str, Any]):
    """Export a comprehensive rankings report."""
    
    try:
        # Create comprehensive report data
        report_sections = []
        
        # Summary
        stats = rankings_data['statistics']
        summary = f"""
        Country Rankings Analysis Report
        
        Analysis Configuration:
        - Metric: {config['metric']}
        - Aggregation: {config['aggregation']}
        - Year: {config.get('year_filter', 'All years')}
        - Countries analyzed: {stats['total_countries']}
        
        Key Statistics:
        - Global average: {stats['global_mean']:.2f}
        - Top 10 average: {stats['top10_mean']:.2f}
        - Bottom 10 average: {stats['bottom10_mean']:.2f}
        - Performance gap: {stats['top10_to_bottom10_ratio']:.1f}x
        """
        report_sections.append(summary)
        
        # Top performers
        top_performers_text = "\nTop Performers:\n"
        for i, (_, row) in enumerate(rankings_data['top_performers'].iterrows(), 1):
            top_performers_text += f"{i}. {row['Area']}: {row[config['metric']]:.2f}\n"
        report_sections.append(top_performers_text)
        
        # Bottom performers
        bottom_performers_text = "\nBottom Performers:\n"
        for i, (_, row) in enumerate(rankings_data['bottom_performers'].iterrows(), 1):
            bottom_performers_text += f"{i}. {row['Area']}: {row[config['metric']]:.2f}\n"
        report_sections.append(bottom_performers_text)
        
        # Combine all sections
        full_report = "\n".join(report_sections)
        
        # Offer download
        st.download_button(
            label="📄 Download Rankings Report",
            data=full_report,
            file_name=f"country_rankings_report_{config['metric']}.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error creating rankings report: {str(e)}")
        st.error(f"Error creating report: {str(e)}")

def render_performance_indicators(rankings_data: Dict[str, Any], config: Dict[str, Any]):
    """Render additional performance indicators and analysis."""
    
    st.markdown("#### 📊 Performance Indicators")
    
    stats = rankings_data['statistics']
    
    # Performance distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribution Analysis:**")
        
        # Calculate quartiles
        all_values = rankings_data['all_countries'][config['metric']]
        q1 = all_values.quantile(0.25)
        q2 = all_values.quantile(0.50)  # median
        q3 = all_values.quantile(0.75)
        
        st.text(f"Q1 (25th percentile): {q1:.2f}")
        st.text(f"Q2 (Median): {q2:.2f}")
        st.text(f"Q3 (75th percentile): {q3:.2f}")
        st.text(f"Interquartile Range: {q3-q1:.2f}")
    
    with col2:
        st.markdown("**Performance Gaps:**")
        
        # Calculate various gaps
        top_1_value = rankings_data['top_performers'].iloc[0][config['metric']]
        median_value = stats['global_median']
        
        st.text(f"Leader advantage: {(top_1_value/median_value):.1f}x median")
        st.text(f"Top 10 vs Median: {(stats['top10_mean']/median_value):.1f}x")
        st.text(f"Coefficient of Variation: {(stats['global_std']/stats['global_mean']):.2f}")

def render_rankings_methodology(config: Dict[str, Any]):
    """Render explanation of rankings methodology."""
    
    with st.expander("ℹ️ Rankings Methodology"):
        st.markdown(f"""
        **Analysis Configuration:**
        - **Metric:** {config['metric']}
        - **Aggregation Method:** {config['aggregation']}
        - **Year Filter:** {config.get('year_filter', 'All available years')}
        
        **Methodology:**
        1. Data is filtered for the specified time period
        2. Values are aggregated by country using the {config['aggregation']} method
        3. Countries are ranked in descending order by the aggregated values
        4. Top and bottom performers are identified
        5. Summary statistics are calculated for comparative analysis
        
        **Interpretation:**
        - Rankings reflect relative performance based on the selected metric
        - Aggregation method affects how multiple data points per country are combined
        - Year filtering allows for period-specific analysis
        - Statistical measures help understand the distribution and gaps between countries
        """)