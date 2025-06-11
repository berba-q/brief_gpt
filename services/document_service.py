"""
Document service for generating Word documents and coordinating document generation.

This module provides functionality to create Word documents and coordinate
between different document generation services (PDF, Word, etc.).
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.shared import OxmlElement, qn
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

from config.config_manager import ConfigManager
from models.data_models import AnalysisResults, VisualizationInfo
from services.pdf_service import PDFService

# Set up logger
logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service for coordinating document generation and creating Word documents.
    
    This service coordinates between different document generation services
    and provides Word document generation capabilities.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the document service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.output_dir = os.path.join(config_manager.get_output_directory(), "reports")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize PDF service
        self.pdf_service = PDFService(self.output_dir)
        
        logger.info(f"Initialized document service with output directory: {self.output_dir}")
    
    def generate_brief(
        self,
        analysis_results: AnalysisResults,
        insights: Dict[str, Any],
        format_type: str = "pdf",
        title: Optional[str] = None,
        author: str = "FAOSTAT Analytics System"
    ) -> str:
        """
        Generate an analytical brief in the specified format.
        
        Args:
            analysis_results: Results from data analysis
            insights: AI-generated insights and narrative
            format_type: Format type ('pdf' or 'docx')
            title: Custom title for the brief
            author: Author name for the document
            
        Returns:
            str: Path to the generated document
        """
        if format_type.lower() == "pdf":
            return self.pdf_service.generate_brief(analysis_results, insights, title, author)
        elif format_type.lower() == "docx":
            return self.generate_word_brief(analysis_results, insights, title, author)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def generate_word_brief(
        self,
        analysis_results: AnalysisResults,
        insights: Dict[str, Any],
        title: Optional[str] = None,
        author: str = "FAOSTAT Analytics System"
    ) -> str:
        """
        Generate a Word document analytical brief.
        
        Args:
            analysis_results: Results from data analysis
            insights: AI-generated insights and narrative
            title: Custom title for the brief
            author: Author name for the document
            
        Returns:
            str: Path to the generated Word document
        """
        if not PYTHON_DOCX_AVAILABLE:
            raise ImportError("python-docx is required for Word document generation")
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = analysis_results.dataset_name.replace(' ', '_').replace('/', '_')
        filename = f"FAOSTAT_Brief_{dataset_name}_{timestamp}.docx"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create document
        doc = Document()
        
        # Set up styles
        self._setup_word_styles(doc)
        
        # Add content
        self._add_word_title_page(doc, analysis_results, title, author)
        self._add_word_highlights(doc, insights)
        self._add_word_background(doc, insights, analysis_results)
        self._add_word_global_trends(doc, insights, analysis_results)
        self._add_word_regional_analysis(doc, insights, analysis_results)
        self._add_word_figures(doc, analysis_results.figures)
        self._add_word_explanatory_notes(doc, insights)
        
        # Save document
        doc.save(filepath)
        
        logger.info(f"Generated Word brief: {filepath}")
        return filepath
    
    def _setup_word_styles(self, doc: Document):
        """Set up custom styles for the Word document."""
        styles = doc.styles
        
        # Title style
        if 'FAO Title' not in [style.name for style in styles]:
            title_style = styles.add_style('FAO Title', WD_STYLE_TYPE.PARAGRAPH)
            title_font = title_style.font
            title_font.name = 'Calibri'
            title_font.size = Pt(20)
            title_font.bold = True
            title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_style.paragraph_format.space_after = Pt(12)
        
        # Heading 1 style
        if 'FAO Heading 1' not in [style.name for style in styles]:
            h1_style = styles.add_style('FAO Heading 1', WD_STYLE_TYPE.PARAGRAPH)
            h1_font = h1_style.font
            h1_font.name = 'Calibri'
            h1_font.size = Pt(16)
            h1_font.bold = True
            h1_style.paragraph_format.space_before = Pt(12)
            h1_style.paragraph_format.space_after = Pt(6)
        
        # Heading 2 style
        if 'FAO Heading 2' not in [style.name for style in styles]:
            h2_style = styles.add_style('FAO Heading 2', WD_STYLE_TYPE.PARAGRAPH)
            h2_font = h2_style.font
            h2_font.name = 'Calibri'
            h2_font.size = Pt(14)
            h2_font.bold = True
            h2_style.paragraph_format.space_before = Pt(10)
            h2_style.paragraph_format.space_after = Pt(4)
        
        # Highlight style
        if 'FAO Highlight' not in [style.name for style in styles]:
            highlight_style = styles.add_style('FAO Highlight', WD_STYLE_TYPE.PARAGRAPH)
            highlight_font = highlight_style.font
            highlight_font.name = 'Calibri'
            highlight_font.size = Pt(11)
            highlight_style.paragraph_format.left_indent = Inches(0.25)
            highlight_style.paragraph_format.space_after = Pt(6)
    
    def _add_word_title_page(
        self,
        doc: Document,
        analysis_results: AnalysisResults,
        title: Optional[str],
        author: str
    ):
        """Add title page to Word document."""
        # Title
        if title:
            brief_title = title
        else:
            brief_title = f"FAOSTAT {analysis_results.dataset_name} Analytical Brief"
        
        title_para = doc.add_paragraph(brief_title, style='FAO Title')
        
        # Subtitle with date range
        if 'year_range' in analysis_results.summary_stats:
            year_range = analysis_results.summary_stats['year_range']
            subtitle = f"Analysis of data from {year_range[0]} to {year_range[1]}"
            subtitle_para = doc.add_paragraph(subtitle)
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add spacing
        doc.add_paragraph()
        doc.add_paragraph()
        
        # Key statistics
        if analysis_results.summary_stats:
            stats_para = doc.add_paragraph()
            stats_para.add_run("Key Dataset Information:").bold = True
            
            if 'row_count' in analysis_results.summary_stats:
                doc.add_paragraph(f"• Total Records: {analysis_results.summary_stats['row_count']:,}")
            
            if 'area_count' in analysis_results.summary_stats:
                doc.add_paragraph(f"• Countries/Territories: {analysis_results.summary_stats['area_count']}")
            
            if 'item_count' in analysis_results.summary_stats:
                doc.add_paragraph(f"• Items Covered: {analysis_results.summary_stats['item_count']}")
        
        # Add page break
        doc.add_page_break()
    
    def _add_word_highlights(self, doc: Document, insights: Dict[str, Any]):
        """Add highlights section to Word document."""
        doc.add_paragraph("HIGHLIGHTS", style='FAO Heading 1')
        
        highlights = insights.get('highlights', [])
        for highlight in highlights:
            para = doc.add_paragraph(f"• {highlight}", style='FAO Highlight')
    
    def _add_word_background(
        self,
        doc: Document,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ):
        """Add background section to Word document."""
        doc.add_paragraph("BACKGROUND", style='FAO Heading 1')
        
        background_text = insights.get('background', '')
        if background_text:
            doc.add_paragraph(background_text)
        
        # Add dataset information
        dataset_info = (
            f"This brief analyzes data from the FAOSTAT {analysis_results.dataset_name} dataset, "
            f"covering {analysis_results.summary_stats.get('row_count', 'N/A')} records across "
            f"{analysis_results.summary_stats.get('area_count', 'N/A')} countries and territories."
        )
        doc.add_paragraph(dataset_info)
    
    def _add_word_global_trends(
        self,
        doc: Document,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ):
        """Add global trends section to Word document."""
        doc.add_paragraph("GLOBAL TRENDS", style='FAO Heading 1')
        
        global_trends = insights.get('global_trends', '')
        if global_trends:
            doc.add_paragraph(global_trends)
        
        # Add time series information if available
        if 'time_series' in analysis_results.summary_stats:
            ts_info = analysis_results.summary_stats['time_series']
            trend_text = (
                f"The data shows a {ts_info.get('direction', 'change')} from "
                f"{ts_info.get('first_year', 'start')} to {ts_info.get('last_year', 'end')}, "
                f"with a total change of {ts_info.get('percent_change', 0):.1f}%."
            )
            doc.add_paragraph(trend_text)
    
    def _add_word_regional_analysis(
        self,
        doc: Document,
        insights: Dict[str, Any],
        analysis_results: AnalysisResults
    ):
        """Add regional analysis section to Word document."""
        doc.add_paragraph("REGIONAL ANALYSIS", style='FAO Heading 1')
        
        regional_analysis = insights.get('regional_analysis', '')
        if regional_analysis:
            doc.add_paragraph(regional_analysis)
        
        # Add top regions information if available
        if 'top_areas' in analysis_results.summary_stats:
            top_areas = analysis_results.summary_stats['top_areas']
            if top_areas:
                areas_text = "The leading countries/regions by volume are: " + \
                           ", ".join(list(top_areas.keys())[:5])
                doc.add_paragraph(areas_text)
    
    def _add_word_figures(self, doc: Document, figures: List[VisualizationInfo]):
        """Add figures section to Word document."""
        if not figures:
            return
        
        doc.add_paragraph("FIGURES", style='FAO Heading 1')
        
        for i, figure in enumerate(figures, 1):
            if os.path.exists(figure.filepath):
                try:
                    # Add figure
                    para = doc.add_paragraph()
                    run = para.runs[0] if para.runs else para.add_run()
                    run.add_picture(figure.filepath, width=Inches(6))
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Add caption
                    caption_para = doc.add_paragraph(f"Figure {i}. {figure.title}")
                    caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    caption_para.runs[0].bold = True
                    
                    # Add description
                    if figure.description:
                        doc.add_paragraph(figure.description)
                    
                    # Add source
                    source_text = (
                        f"Source: FAO. FAOSTAT. Rome. {datetime.now().strftime('%Y')}. "
                        "http://www.fao.org/faostat"
                    )
                    source_para = doc.add_paragraph(source_text)
                    source_para.runs[0].italic = True
                    source_para.runs[0].font.size = Pt(9)
                    
                    doc.add_paragraph()  # Add spacing
                    
                except Exception as e:
                    logger.warning(f"Could not include figure {figure.filepath}: {str(e)}")
    
    def _add_word_explanatory_notes(self, doc: Document, insights: Dict[str, Any]):
        """Add explanatory notes section to Word document."""
        doc.add_paragraph("EXPLANATORY NOTES", style='FAO Heading 1')
        
        explanatory_notes = insights.get('explanatory_notes', '')
        if explanatory_notes:
            doc.add_paragraph(explanatory_notes)
        
        # Add standard data source note
        data_note = (
            "Data sources: The main data source for this analysis is the FAOSTAT database "
            "maintained by the Food and Agriculture Organization of the United Nations (FAO). "
            "Data are updated annually and may be subject to revision as new information becomes available."
        )
        doc.add_paragraph(data_note)
    
    def get_available_formats(self) -> List[str]:
        """Get list of available document formats."""
        formats = []
        
        if self.pdf_service.is_available():
            formats.append("pdf")
        
        if PYTHON_DOCX_AVAILABLE:
            formats.append("docx")
        
        return formats
    
    def is_format_available(self, format_type: str) -> bool:
        """Check if a specific format is available."""
        return format_type.lower() in self.get_available_formats()