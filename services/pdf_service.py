"""
PDF service for generating professional analytical briefs.

This module provides functionality to create PDF reports in the style of
official FAO analytical briefs, including proper formatting, charts, and
structured content.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor, blue, black, grey
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics.charts.lineplots import LinePlot
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from models.data_models import AnalysisResults, VisualizationInfo

# Set up logger
logger = logging.getLogger(__name__)

class PDFService:
    """
    Service for generating PDF analytical briefs.
    
    Creates professional PDF reports following FAO analytical brief
    formatting standards with proper layout, typography, and chart integration.
    """
    
    def __init__(self, output_dir: str = "output/reports"):
        """
        Initialize the PDF service.
        
        Args:
            output_dir: Directory to save generated PDF reports
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if ReportLab is available
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. PDF generation will be disabled.")
            return
        
        # Define FAO color scheme
        self.colors = {
            'fao_blue': HexColor('#0066CC'),
            'fao_green': HexColor('#009639'),
            'fao_orange': HexColor('#FF6600'),
            'dark_grey': HexColor('#333333'),
            'light_grey': HexColor('#F5F5F5'),
            'medium_grey': HexColor('#999999')
        }
        
        # Set up styles
        self._setup_styles()
        
        logger.info(f"Initialized PDF service with output directory: {output_dir}")
    
    def _setup_styles(self):
        """Set up custom paragraph styles for the document."""
        if not REPORTLAB_AVAILABLE:
            return
        
        # Get base styles
        styles = getSampleStyleSheet()
        
        # Define custom styles
        self.styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=0.3*inch,
                textColor=self.colors['fao_blue'],
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ),
            'heading1': ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceBefore=0.2*inch,
                spaceAfter=0.1*inch,
                textColor=self.colors['fao_blue'],
                fontName='Helvetica-Bold'
            ),
            'heading2': ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=0.15*inch,
                spaceAfter=0.08*inch,
                textColor=self.colors['dark_grey'],
                fontName='Helvetica-Bold'
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceBefore=0.05*inch,
                spaceAfter=0.05*inch,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            ),
            'highlight': ParagraphStyle(
                'CustomHighlight',
                parent=styles['Normal'],
                fontSize=10,
                leftIndent=0.2*inch,
                rightIndent=0.2*inch,
                spaceBefore=0.05*inch,
                spaceAfter=0.05*inch,
                borderColor=self.colors['fao_green'],
                borderWidth=1,
                borderPadding=0.1*inch,
                backColor=self.colors['light_grey']
            ),
            'caption': ParagraphStyle(
                'CustomCaption',
                parent=styles['Normal'],
                fontSize=9,
                alignment=TA_CENTER,
                textColor=self.colors['medium_grey'],
                fontName='Helvetica-Oblique'
            ),
            'source': ParagraphStyle(
                'CustomSource',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_LEFT,
                textColor=self.colors['medium_grey'],
                fontName='Helvetica'
            )
        }
    
    def generate_brief(
        self,
        analysis_results: AnalysisResults,
        insights: Dict[str, Any],
        title: Optional[str] = None,
        author: str = "FAOSTAT Analytics System"
    ) -> str:
        """
        Generate a complete analytical brief PDF.
        
        Args:
            analysis_results: Results from data analysis
            insights: AI-generated insights and narrative
            title: Custom title for the brief
            author: Author name for the document
            
        Returns:
            str: Path to the generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = analysis_results.dataset_name.replace(' ', '_').replace('/', '_')
        filename = f"FAOSTAT_Brief_{dataset_name}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Build content
        story = []
        
        # Add cover page
        story.extend(self._build_cover_page(analysis_results, title, author))
        story.append(PageBreak())
        
        # Add highlights section
        story.extend(self._build_highlights_section(insights))
        
        # Add background section
        story.extend(self._build_background_section(insights, analysis_results))
        
        # Add global trends section
        story.extend(self._build_global_trends_section(insights, analysis_results))
        
        # Add regional analysis section
        story.extend(self._build_regional_analysis_section(insights, analysis_results))
        
        # Add figures
        story.extend(self._build_figures_section(analysis_results.figures))
        
        # Add explanatory notes
        story.extend(self._build_explanatory_notes_section(insights))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated PDF brief: {filepath}")
        return filepath
    
    def _build_cover_page(
        self,
        analysis_results: AnalysisResults,
        title: Optional[str],
        author: str
    ) -> List:
        """Build the cover page of the brief."""
        story = []
        
        # FAO header (if logo available)
        fao_logo_path = "static/images/fao_logo.png"
        if os.path.exists(fao_logo_path):
            logo = Image(fao_logo_path, width=2*inch, height=0.8*inch)
            story.append(logo)
            story.append(Spacer(1, 0.5*inch))
        
        # Title
        if title:
            brief_title = title
        else:
            brief_title = f"FAOSTAT {analysis_results.dataset_name} Analytical Brief"
        
        story.append(Paragraph(brief_title, self.styles['title']))
        story.append(Spacer(1, 0.3*inch))
        
        # Subtitle with date range
        if 'year_range' in analysis_results.summary_stats:
            year_range = analysis_results.summary_stats['year_range']
            subtitle = f"Analysis of data from {year_range[0]} to {year_range[1]}"
            story.append(Paragraph(subtitle, self.styles['body']))
        
        story.append(Spacer(1, 1*inch))
        
        # Key statistics box
        if analysis_results.summary_stats:
            stats_data = self._format_key_statistics(analysis_results.summary_stats)
            if stats_data:
                stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors['fao_blue']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), ['white', self.colors['light_grey']])
                ]))
                story.append(stats_table)
        
        story.append(Spacer(1, 2*inch))
        
        # Footer with date and author
        footer_text = f"Generated on {datetime.now().strftime('%B %d, %Y')}<br/>By: {author}"
        story.append(Paragraph(footer_text, self.styles['caption']))
        
        return story
    
    def _build_highlights_section(self, insights: Dict[str, Any]) -> List:
        """Build the highlights section."""
        story = []
        
        story.append(Paragraph("HIGHLIGHTS", self.styles['heading1']))
        
        highlights = insights.get('highlights', [])
        for highlight in highlights:
            bullet_text = f"• {highlight}"
            story.append(Paragraph(bullet_text, self.styles['highlight']))
        
        story.append(Spacer(1, 0.2*inch))
        return story
    
    def _build_background_section(
        self,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ) -> List:
        """Build the background section."""
        story = []
        
        story.append(Paragraph("BACKGROUND", self.styles['heading1']))
        
        background_text = insights.get('background', '')
        if background_text:
            story.append(Paragraph(background_text, self.styles['body']))
        
        # Add dataset information
        dataset_info = f"""
        This brief analyzes data from the FAOSTAT {analysis_results.dataset_name} dataset, 
        covering {analysis_results.summary_stats.get('row_count', 'N/A')} records across 
        {analysis_results.summary_stats.get('area_count', 'N/A')} countries and territories.
        """
        story.append(Paragraph(dataset_info, self.styles['body']))
        
        story.append(Spacer(1, 0.15*inch))
        return story
    
    def _build_global_trends_section(
        self,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ) -> List:
        """Build the global trends section."""
        story = []
        
        story.append(Paragraph("GLOBAL TRENDS", self.styles['heading1']))
        
        global_trends = insights.get('global_trends', '')
        if global_trends:
            story.append(Paragraph(global_trends, self.styles['body']))
        
        # Add time series information if available
        if 'time_series' in analysis_results.summary_stats:
            ts_info = analysis_results.summary_stats['time_series']
            trend_text = f"""
            The data shows a {ts_info.get('direction', 'change')} from 
            {ts_info.get('first_year', 'start')} to {ts_info.get('last_year', 'end')}, 
            with a total change of {ts_info.get('percent_change', 0):.1f}%.
            """
            story.append(Paragraph(trend_text, self.styles['body']))
        
        story.append(Spacer(1, 0.15*inch))
        return story
    
    def _build_regional_analysis_section(
        self,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ) -> List:
        """Build the regional analysis section."""
        story = []
        
        story.append(Paragraph("REGIONAL ANALYSIS", self.styles['heading1']))
        
        regional_analysis = insights.get('regional_analysis', '')
        if regional_analysis:
            story.append(Paragraph(regional_analysis, self.styles['body']))
        
        # Add top regions information if available
        if 'top_areas' in analysis_results.summary_stats:
            top_areas = analysis_results.summary_stats['top_areas']
            if top_areas:
                areas_text = "The leading countries/regions by volume are: " + \
                           ", ".join(list(top_areas.keys())[:5])
                story.append(Paragraph(areas_text, self.styles['body']))
        
        story.append(Spacer(1, 0.15*inch))
        return story
    
    def _build_figures_section(self, figures: List[VisualizationInfo]) -> List:
        """Build the figures section with charts."""
        story = []
        
        if not figures:
            return story
        
        story.append(Paragraph("FIGURES", self.styles['heading1']))
        
        for i, figure in enumerate(figures, 1):
            if os.path.exists(figure.filepath):
                # Add figure
                try:
                    img = Image(figure.filepath, width=6*inch, height=4*inch)
                    story.append(img)
                    
                    # Add caption
                    caption = f"Figure {i}. {figure.title}"
                    story.append(Paragraph(caption, self.styles['caption']))
                    
                    # Add description
                    if figure.description:
                        story.append(Paragraph(figure.description, self.styles['body']))
                    
                    # Add source
                    source_text = "Source: FAO. FAOSTAT. Rome. " + \
                                datetime.now().strftime('%Y') + \
                                ". http://www.fao.org/faostat"
                    story.append(Paragraph(source_text, self.styles['source']))
                    
                    story.append(Spacer(1, 0.2*inch))
                    
                except Exception as e:
                    logger.warning(f"Could not include figure {figure.filepath}: {str(e)}")
        
        return story
    
    def _build_explanatory_notes_section(self, insights: Dict[str, Any]) -> List:
        """Build the explanatory notes section."""
        story = []
        
        story.append(Paragraph("EXPLANATORY NOTES", self.styles['heading1']))
        
        explanatory_notes = insights.get('explanatory_notes', '')
        if explanatory_notes:
            story.append(Paragraph(explanatory_notes, self.styles['body']))
        
        # Add standard data source note
        data_note = """
        Data sources: The main data source for this analysis is the FAOSTAT database 
        maintained by the Food and Agriculture Organization of the United Nations (FAO). 
        Data are updated annually and may be subject to revision as new information becomes available.
        """
        story.append(Paragraph(data_note, self.styles['body']))
        
        return story
    
    def _format_key_statistics(self, summary_stats: Dict[str, Any]) -> List[List[str]]:
        """Format key statistics for the cover page table."""
        data = [['Indicator', 'Value']]
        
        if 'row_count' in summary_stats:
            data.append(['Total Records', f"{summary_stats['row_count']:,}"])
        
        if 'area_count' in summary_stats:
            data.append(['Countries/Territories', str(summary_stats['area_count'])])
        
        if 'year_range' in summary_stats:
            year_range = summary_stats['year_range']
            data.append(['Time Period', f"{year_range[0]} - {year_range[1]}"])
        
        if 'item_count' in summary_stats:
            data.append(['Items Covered', str(summary_stats['item_count'])])
        
        return data if len(data) > 1 else None
    
    def is_available(self) -> bool:
        """Check if PDF generation is available."""
        return REPORTLAB_AVAILABLE