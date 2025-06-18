"""
Thematic templates page for the FAOSTAT Analytics application.

This module provides pre-configured analysis templates for common agricultural
policy questions and research topics.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List, Optional

from config.constants import THEMATIC_TEMPLATES
from models.data_models import AnalysisResults, AnalysisFilter, VisualizationInfo
from features.template_manager import TemplateManager

logger = logging.getLogger(__name__)

def show_thematic_templates_page():
    """Render the thematic templates page."""
    
    st.title("📋 Thematic Analysis Templates")
    st.markdown("Professional analytical frameworks for common agricultural policy questions")
    
    # Initialize template manager
    template_manager = TemplateManager()
    
    # Template selection and overview
    render_template_overview()
    
    # Selected template details
    selected_template = st.session_state.get('selected_template')
    
    if selected_template:
        render_template_execution(selected_template, template_manager)
    else:
        render_template_selection()

def render_template_overview():
    """Render overview of available templates."""
    
    st.markdown("## 🌟 Available Templates")
    
    # Template categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Templates", len(THEMATIC_TEMPLATES))
    
    with col2:
        st.metric("Policy Areas", "4")  # Food Security, Climate, Trade, Fertilizers
    
    with col3:
        # Count total datasets covered
        all_datasets = set()
        for template in THEMATIC_TEMPLATES.values():
            for dataset in template['datasets']:
                all_datasets.add(dataset['code'])
        st.metric("Datasets Covered", len(all_datasets))

def render_template_selection():
    """Render template selection interface."""
    
    st.markdown("## 🎯 Choose Your Analysis Template")
    
    # Display templates in cards
    for template_key, template in THEMATIC_TEMPLATES.items():
        render_template_card(template_key, template)

def render_template_card(template_key: str, template: Dict[str, Any]):
    """Render a single template card."""
    
    with st.container():
        # Create bordered container
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin: 1rem 0; background-color: #f8f9fa;">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📊 {template['name']}")
            st.markdown(f"_{template['description']}_")
            
            # Show key features
            st.markdown("**Key Features:**")
            
            # Datasets
            dataset_names = [ds['name'] for ds in template['datasets'][:2]]
            if len(template['datasets']) > 2:
                dataset_names.append(f"+ {len(template['datasets']) - 2} more")
            st.markdown(f"• **Datasets:** {', '.join(dataset_names)}")
            
            # Focus areas
            if template.get('regions_focus'):
                regions_text = ', '.join(template['regions_focus'][:3])
                if len(template['regions_focus']) > 3:
                    regions_text += f" + {len(template['regions_focus']) - 3} more"
                st.markdown(f"• **Focus Regions:** {regions_text}")
            
            # Key indicators
            indicators_text = ', '.join(template['key_indicators'][:3])
            if len(template['key_indicators']) > 3:
                indicators_text += "..."
            st.markdown(f"• **Key Indicators:** {indicators_text}")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(f"🚀 Use Template", key=f"select_{template_key}", use_container_width=True):
                st.session_state.selected_template = template_key
                st.rerun()
            
            if st.button(f"👀 Preview", key=f"preview_{template_key}", use_container_width=True):
                show_template_preview(template_key, template)
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_template_preview(template_key: str, template: Dict[str, Any]):
    """Show detailed template preview."""
    
    st.markdown("---")
    st.markdown(f"## 👀 Preview: {template['name']}")
    
    # Template details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Datasets Required")
        for dataset in template['datasets']:
            st.markdown(f"• **{dataset['name']}** (`{dataset['code']}`)")
        
        st.markdown("### 🎯 Key Indicators")
        for indicator in template['key_indicators']:
            st.markdown(f"• {indicator}")
    
    with col2:
        st.markdown("### 🌍 Focus Regions")
        for region in template['regions_focus']:
            st.markdown(f"• {region}")
        
        st.markdown("### 📊 Default Visualizations")
        for viz in template['default_visualizations']:
            st.markdown(f"• {viz.replace('_', ' ').title()}")
    
    # Analysis framework
    st.markdown("### 🔬 Analysis Framework")
    st.markdown(template['prompt_template'])
    
    # Close preview
    if st.button("❌ Close Preview"):
        st.rerun()

def render_template_execution(template_key: str, template_manager):
    """Render template execution interface."""
    
    template = THEMATIC_TEMPLATES[template_key]
    
    st.markdown(f"## 🔬 Executing: {template['name']}")
    st.markdown(f"_{template['description']}_")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Template execution steps
    steps = [
        "📊 Loading datasets",
        "🔧 Configuring analysis",
        "📈 Generating visualizations", 
        "🤖 Creating AI insights",
        "📄 Compiling report"
    ]
    
    # Step 1: Dataset loading
    status_text.text("Step 1/5: Loading required datasets...")
    progress_bar.progress(0.2)
    
    datasets_status = load_template_datasets(template)
    
    if not datasets_status['success']:
        st.error("❌ Failed to load required datasets")
        render_dataset_loading_issues(datasets_status)
        return
    
    # Step 2: Configuration
    status_text.text("Step 2/5: Configuring analysis parameters...")
    progress_bar.progress(0.4)
    
    config = render_template_configuration(template, datasets_status['loaded_datasets'])
    
    # Step 3: Analysis execution
    if st.button("▶️ Run Analysis", type="primary"):
        execute_template_analysis(template, config, datasets_status['loaded_datasets'], progress_bar, status_text)

def load_template_datasets(template: Dict[str, Any]) -> Dict[str, Any]:
    """Load datasets required by the template."""
    
    faostat_service = st.session_state.get('faostat_service')
    
    if not faostat_service:
        return {'success': False, 'error': 'FAOSTAT service not available'}
    
    loaded_datasets = {}
    failed_datasets = []
    
    for dataset_info in template['datasets']:
        dataset_code = dataset_info['code']
        
        try:
            with st.spinner(f"Loading {dataset_info['name']}..."):
                df, metadata = faostat_service.get_dataset(dataset_code)
                
                if df is not None and not df.empty:
                    loaded_datasets[dataset_code] = {
                        'data': df,
                        'metadata': metadata,
                        'name': dataset_info['name']
                    }
                    st.success(f"✅ Loaded {dataset_info['name']} ({len(df):,} records)")
                else:
                    failed_datasets.append(dataset_info)
                    st.warning(f"⚠️ Could not load {dataset_info['name']}")
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_code}: {str(e)}")
            failed_datasets.append(dataset_info)
            st.error(f"❌ Error loading {dataset_info['name']}: {str(e)}")
    
    return {
        'success': len(loaded_datasets) > 0,
        'loaded_datasets': loaded_datasets,
        'failed_datasets': failed_datasets
    }

def render_dataset_loading_issues(datasets_status: Dict[str, Any]):
    """Render information about dataset loading issues."""
    
    st.markdown("### ⚠️ Dataset Loading Issues")
    
    if datasets_status.get('failed_datasets'):
        st.markdown("**Failed to load:**")
        for dataset in datasets_status['failed_datasets']:
            st.markdown(f"• {dataset['name']} (`{dataset['code']}`)")
    
    st.markdown("**Options:**")
    st.markdown("1. Try refreshing the datasets")
    st.markdown("2. Check your internet connection")
    st.markdown("3. Select alternative datasets")
    st.markdown("4. Continue with available datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retry Loading"):
            st.rerun()
    
    with col2:
        if st.button("➡️ Continue Anyway"):
            # Could implement partial template execution
            st.info("Partial analysis not yet implemented")

def render_template_configuration(
    template: Dict[str, Any], 
    loaded_datasets: Dict[str, Any]
) -> Dict[str, Any]:
    """Render template configuration interface."""
    
    st.markdown("### ⚙️ Analysis Configuration")
    
    config = {}
    
    # Time period configuration
    with st.expander("📅 Time Period Settings", expanded=True):
        
        # Find available year ranges across datasets
        year_ranges = []
        for dataset_code, dataset_info in loaded_datasets.items():
            df = dataset_info['data']
            if 'Year' in df.columns:
                min_year = int(df['Year'].min())
                max_year = int(df['Year'].max())
                year_ranges.append((min_year, max_year))
        
        if year_ranges:
            overall_min = min(r[0] for r in year_ranges)
            overall_max = max(r[1] for r in year_ranges)
            
            config['year_range'] = st.slider(
                "Analysis period",
                min_value=overall_min,
                max_value=overall_max,
                value=(max(overall_min, overall_max - 10), overall_max),
                help=f"Data available from {overall_min} to {overall_max}"
            )
        else:
            config['year_range'] = None
            st.info("No time series data available")
    
    # Geographic focus
    with st.expander("🌍 Geographic Focus", expanded=True):
        
        # Get available regions from template
        template_regions = template.get('regions_focus', [])
        
        region_option = st.radio(
            "Regional scope",
            ["Template default", "Custom selection", "Global analysis"]
        )
        
        if region_option == "Template default":
            config['regions'] = template_regions
            st.info(f"Using template regions: {', '.join(template_regions)}")
        
        elif region_option == "Custom selection":
            # Get available countries from datasets
            all_countries = set()
            for dataset_info in loaded_datasets.values():
                df = dataset_info['data']
                if 'Area' in df.columns:
                    all_countries.update(df['Area'].unique())
            
            config['regions'] = st.multiselect(
                "Select countries/regions",
                options=sorted(all_countries),
                default=[c for c in template_regions if c in all_countries]
            )
        
        else:  # Global analysis
            config['regions'] = None
            st.info("Global analysis - all countries included")
    
    # Analysis focus
    with st.expander("🔬 Analysis Focus", expanded=True):
        
        config['key_indicators'] = st.multiselect(
            "Key indicators to emphasize",
            options=template['key_indicators'],
            default=template['key_indicators'][:3]
        )
        
        config['analysis_depth'] = st.selectbox(
            "Analysis depth",
            options=["Executive Summary", "Standard Analysis", "Comprehensive Deep Dive"],
            index=1
        )
        
        config['include_comparisons'] = st.checkbox(
            "Include comparative analysis",
            value=True,
            help="Compare between regions, time periods, or indicators"
        )
    
    return config

def execute_template_analysis(
    template: Dict[str, Any],
    config: Dict[str, Any],
    loaded_datasets: Dict[str, Any],
    progress_bar,
    status_text
):
    """Execute the template analysis."""
    
    try:
        # Step 3: Generate visualizations
        status_text.text("Step 3/5: Generating visualizations...")
        progress_bar.progress(0.6)
        
        all_figures = []
        
        for dataset_code, dataset_info in loaded_datasets.items():
            df = dataset_info['data']
            
            # Apply filters based on configuration
            filtered_df = apply_template_filters(df, config)
            
            if not filtered_df.empty:
                # Generate visualizations
                viz_generator = st.session_state.get('viz_generator')
                if viz_generator:
                    figures = viz_generator.generate_visualizations(filtered_df, dataset_info['name'])
                    all_figures.extend(figures)
        
        # Step 4: Generate AI insights
        status_text.text("Step 4/5: Generating AI insights...")
        progress_bar.progress(0.8)
        
        insights = generate_template_insights(template, config, loaded_datasets, all_figures)
        
        # Step 5: Compile results
        status_text.text("Step 5/5: Compiling final report...")
        progress_bar.progress(1.0)
        
        # Create analysis results
        analysis_results = compile_template_results(template, config, loaded_datasets, all_figures, insights)
        
        # Display results
        status_text.text("✅ Analysis complete!")
        display_template_results(template, analysis_results, insights)
        
    except Exception as e:
        logger.error(f"Error executing template analysis: {str(e)}")
        st.error(f"Error during analysis: {str(e)}")

def apply_template_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply configuration filters to dataset."""
    
    filtered_df = df.copy()
    
    # Apply year filter
    if config.get('year_range') and 'Year' in filtered_df.columns:
        start_year, end_year = config['year_range']
        filtered_df = filtered_df[
            (filtered_df['Year'] >= start_year) & 
            (filtered_df['Year'] <= end_year)
        ]
    
    # Apply region filter
    if config.get('regions') and 'Area' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Area'].isin(config['regions'])]
    
    return filtered_df

def generate_template_insights(
    template: Dict[str, Any],
    config: Dict[str, Any], 
    loaded_datasets: Dict[str, Any],
    figures: List[VisualizationInfo]
) -> Dict[str, Any]:
    """Generate AI insights using the template framework."""
    
    openai_service = st.session_state.get('openai_service')
    
    if not openai_service:
        return {
            'highlights': ['AI insights not available - OpenAI API key not configured'],
            'background': 'Template analysis completed without AI insights.',
            'global_trends': 'Manual analysis required.',
            'regional_analysis': 'Manual analysis required.',
            'explanatory_notes': 'Configure OpenAI API key for AI-powered insights.'
        }
    
    try:
        # Combine summary statistics from all datasets
        combined_stats = {}
        
        for dataset_code, dataset_info in loaded_datasets.items():
            df = apply_template_filters(dataset_info['data'], config)
            
            if not df.empty:
                faostat_service = st.session_state.faostat_service
                stats = faostat_service.generate_summary_stats(df)
                combined_stats[dataset_code] = stats
        
        # Generate insights using template prompt
        figure_descriptions = [fig.description for fig in figures]
        
        # Use template-specific prompt
        from utils.prompt_templates import get_thematic_analysis_prompt
        
        prompt = get_thematic_analysis_prompt(
            template_name=template['name'],
            template_prompt=template['prompt_template'],
            dataset_name=f"Multi-dataset analysis: {', '.join(loaded_datasets.keys())}",
            summary_stats=combined_stats,
            figures_descriptions=figure_descriptions
        )
        
        messages = [
            {"role": "system", "content": "You are an expert agricultural policy analyst creating professional analytical briefs."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai_service._call_openai_api(messages)
        
        # Parse response (assuming structured format)
        # In practice, you might want more sophisticated parsing
        insights = {
            'highlights': [response[:200] + "..."],
            'background': response,
            'global_trends': response,
            'regional_analysis': response,
            'explanatory_notes': 'Analysis based on template framework and available data.'
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating template insights: {str(e)}")
        return {
            'highlights': [f'Error generating insights: {str(e)}'],
            'background': 'Template analysis completed with limited insights.',
            'global_trends': 'Manual analysis required.',
            'regional_analysis': 'Manual analysis required.',
            'explanatory_notes': f'Error in AI processing: {str(e)}'
        }

def compile_template_results(
    template: Dict[str, Any],
    config: Dict[str, Any],
    loaded_datasets: Dict[str, Any],
    figures: List[VisualizationInfo],
    insights: Dict[str, Any]
) -> AnalysisResults:
    """Compile final analysis results."""
    
    # Combine summary statistics
    combined_stats = {
        'template_name': template['name'],
        'datasets_used': list(loaded_datasets.keys()),
        'configuration': config,
        'total_records': sum(len(ds['data']) for ds in loaded_datasets.values()),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Create analysis results object
    analysis_results = AnalysisResults(
        dataset_code="TEMPLATE_" + template['name'].replace(' ', '_').upper(),
        dataset_name=f"Thematic Analysis: {template['name']}",
        summary_stats=combined_stats,
        figures=figures,
        filter_options=AnalysisFilter(
            year_range=config.get('year_range'),
            regions=config.get('regions'),
            items=None,
            elements=None
        ),
        insights=insights
    )
    
    return analysis_results

def display_template_results(
    template: Dict[str, Any],
    analysis_results: AnalysisResults,
    insights: Dict[str, Any]
):
    """Display the template analysis results."""
    
    st.success("🎉 Template analysis completed successfully!")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Executive Summary",
        "📊 Visualizations", 
        "🤖 AI Insights",
        "📄 Generate Report"
    ])
    
    with tab1:
        render_executive_summary(template, analysis_results, insights)
    
    with tab2:
        render_template_visualizations(analysis_results.figures)
    
    with tab3:
        render_template_insights_display(insights)
    
    with tab4:
        render_report_generation(template, analysis_results, insights)

def render_executive_summary(
    template: Dict[str, Any],
    analysis_results: AnalysisResults,
    insights: Dict[str, Any]
):
    """Render executive summary of template results."""
    
    st.markdown(f"## 📋 Executive Summary: {template['name']}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datasets Used", len(analysis_results.summary_stats['datasets_used']))
    
    with col2:
        st.metric("Total Records", f"{analysis_results.summary_stats['total_records']:,}")
    
    with col3:
        st.metric("Visualizations", len(analysis_results.figures))
    
    with col4:
        st.metric("Analysis Type", "Thematic")
    
    # Key highlights
    if insights.get('highlights'):
        st.markdown("### ✨ Key Findings")
        for highlight in insights['highlights']:
            st.info(f"💡 {highlight}")
    
    # Template framework
    st.markdown("### 🔬 Analysis Framework")
    st.markdown(template['description'])
    
    # Datasets used
    st.markdown("### 📊 Data Sources")
    for dataset_code in analysis_results.summary_stats['datasets_used']:
        st.markdown(f"• {dataset_code}")

def render_template_visualizations(figures: List[VisualizationInfo]):
    """Render template visualizations."""
    
    if not figures:
        st.info("No visualizations generated")
        return
    
    from components.visualization_gallery import render_visualization_gallery
    
    render_visualization_gallery(
        figures=figures,
        title="📊 Template Analysis Visualizations",
        columns=2,
        show_download=True,
        show_details=True
    )

def render_template_insights_display(insights: Dict[str, Any]):
    """Display AI insights from template analysis."""
    
    st.markdown("## 🤖 AI-Generated Insights")
    
    # Background
    if insights.get('background'):
        st.markdown("### 📖 Background")
        st.write(insights['background'])
    
    # Global trends
    if insights.get('global_trends'):
        st.markdown("### 🌍 Global Trends")
        st.write(insights['global_trends'])
    
    # Regional analysis
    if insights.get('regional_analysis'):
        st.markdown("### 🗺️ Regional Analysis")
        st.write(insights['regional_analysis'])
    
    # Explanatory notes
    if insights.get('explanatory_notes'):
        st.markdown("### ⚠️ Notes and Limitations")
        st.write(insights['explanatory_notes'])

def render_report_generation(
    template: Dict[str, Any],
    analysis_results: AnalysisResults,
    insights: Dict[str, Any]
):
    """Render report generation interface."""
    
    st.markdown("## 📄 Generate Professional Report")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Report Title",
            value=f"FAOSTAT Analytical Brief: {template['name']}",
            help="Title for the generated report"
        )
        
        author_name = st.text_input(
            "Author",
            value="FAOSTAT Analytics System",
            help="Author name for the report"
        )
    
    with col2:
        report_format = st.selectbox(
            "Report Format",
            options=["PDF", "Word Document"],
            help="Choose the output format"
        )
        
        include_appendix = st.checkbox(
            "Include Data Appendix",
            value=True,
            help="Include detailed data tables in appendix"
        )
    
    # Generate report
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("📄 Generating professional report..."):
            try:
                doc_service = st.session_state.get('doc_service')
                
                if doc_service:
                    format_type = "pdf" if report_format == "PDF" else "docx"
                    
                    report_path = doc_service.generate_brief(
                        analysis_results=analysis_results,
                        insights=insights,
                        format_type=format_type,
                        title=report_title,
                        author=author_name
                    )
                    
                    st.success("✅ Report generated successfully!")
                    st.info(f"📁 Saved to: {report_path}")
                    
                    # Download option
                    with open(report_path, "rb") as file:
                        mime_type = "application/pdf" if format_type == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        st.download_button(
                            label=f"📥 Download {report_format}",
                            data=file.read(),
                            file_name=f"{report_title.replace(' ', '_')}.{format_type}",
                            mime=mime_type
                        )
                else:
                    st.error("❌ Document service not available")
            
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                st.error(f"Error generating report: {str(e)}")
    
    # Template reset
    st.markdown("---")
    if st.button("🔄 Start New Template Analysis"):
        # Clear template selection
        if 'selected_template' in st.session_state:
            del st.session_state.selected_template
        st.rerun()