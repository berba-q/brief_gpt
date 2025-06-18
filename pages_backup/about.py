"""
About page for the FAOSTAT Analytics application.

This module provides information about the application, its features,
data sources, and usage guidelines.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

def show_about_page():
    """Render the about page."""
    st.write("✅ About Page Entry Reached") # Debugging line to confirm page entry
    
    st.title("ℹ️ About FAOSTAT Analytics")
    
    # Overview section
    render_overview()
    
    # Features section
    render_features()
    
    # Data sources section
    render_data_sources()
    
    # Technical information
    render_technical_info()
    
    # Usage guide
    render_usage_guide()
    
    # Credits and attribution
    render_credits()

def render_overview():
    """Render application overview."""
    
    st.markdown("""
    ## 🌾 Overview
    
    **FAOSTAT Analytics** is a comprehensive platform for analyzing Food and Agriculture Organization (FAO) 
    statistical data and generating professional analytical briefs. The application combines robust data 
    processing capabilities with AI-powered insights to create publication-ready reports in the style of 
    official FAO analytical briefs.
    
    ### 🎯 Mission
    
    To democratize access to agricultural data analysis and enable researchers, policymakers, and analysts 
    to generate professional-quality insights from FAOSTAT data quickly and efficiently.
    """)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Datasets Available", "200+")
    
    with col2:
        st.metric("🌍 Countries Covered", "190+")
    
    with col3:
        st.metric("📈 Years of Data", "60+")
    
    with col4:
        st.metric("🤖 AI-Powered", "Yes")

def render_features():
    """Render application features."""
    
    st.markdown("## ✨ Key Features")
    
    features = {
        "📊 Comprehensive Data Access": {
            "description": "Browse and analyze 200+ FAOSTAT datasets covering production, trade, food security, and more",
            "benefits": [
                "Real-time data fetching from FAOSTAT API",
                "Interactive filtering and exploration",
                "Preview capabilities before full analysis",
                "Data quality indicators and metadata"
            ]
        },
        "🤖 AI-Powered Insights": {
            "description": "Generate expert-level analysis using advanced GPT models",
            "benefits": [
                "Natural language query processing",
                "Automated insight generation",
                "Professional narrative creation",
                "Context-aware interpretations"
            ]
        },
        "📈 Professional Visualizations": {
            "description": "Create publication-ready charts and graphs with consistent styling",
            "benefits": [
                "Multiple chart types (time series, comparisons, distributions)",
                "FAO-standard color schemes",
                "High-resolution export options",
                "Interactive visualization gallery"
            ]
        },
        "📄 Automated Report Generation": {
            "description": "Generate PDF and Word documents following FAO formatting standards",
            "benefits": [
                "Professional layout and typography",
                "Structured analytical sections",
                "Integrated charts and tables",
                "Customizable templates"
            ]
        },
        "🔍 Interactive Exploration": {
            "description": "Filter, query, and explore data with intuitive interfaces",
            "benefits": [
                "Multi-dimensional filtering",
                "Real-time data updates",
                "Bookmark favorite analyses",
                "Export filtered datasets"
            ]
        },
        "📋 Thematic Templates": {
            "description": "Pre-configured analysis frameworks for common policy questions",
            "benefits": [
                "Food security analysis",
                "Climate impact assessment",
                "Trade pattern analysis",
                "Fertilizer use evaluation"
            ]
        }
    }
    
    # Display features in expandable sections
    for feature_name, feature_info in features.items():
        with st.expander(f"**{feature_name}**"):
            st.markdown(feature_info["description"])
            st.markdown("**Key Benefits:**")
            for benefit in feature_info["benefits"]:
                st.markdown(f"• {benefit}")

def render_data_sources():
    """Render information about data sources."""
    
    st.markdown("## 📚 Data Sources")
    
    st.markdown("""
    ### 🏛️ FAOSTAT Database
    
    **FAOSTAT** is the Food and Agriculture Organization's comprehensive statistical database, providing 
    free access to food and agriculture data for over 245 countries and territories.
    
    **Coverage includes:**
    """)
    
    data_domains = {
        "🌾 Production": [
            "Crops and livestock products",
            "Production indices",
            "Producer prices"
        ],
        "🔄 Trade": [
            "Import/export values and quantities",
            "Trade matrices",
            "Trade indices"
        ],
        "🍽️ Food Security": [
            "Food balance sheets",
            "Food security indicators",
            "Dietary energy supply"
        ],
        "🌱 Resources": [
            "Land use and agricultural area",
            "Fertilizer consumption",
            "Pesticide use"
        ],
        "🌡️ Environment": [
            "Greenhouse gas emissions",
            "Land use change",
            "Climate data"
        ],
        "👥 Population": [
            "Demographic indicators",
            "Rural/urban population",
            "Economic indicators"
        ]
    }
    
    cols = st.columns(2)
    col_items = list(data_domains.items())
    
    for i, (domain, items) in enumerate(col_items):
        with cols[i % 2]:
            st.markdown(f"**{domain}**")
            for item in items:
                st.markdown(f"• {item}")
    
    # Data quality and updates
    st.markdown("### 📊 Data Quality & Updates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Quality:**
        - Official government statistics
        - Standardized methodologies
        - Quality control procedures
        - Metadata documentation
        """)
    
    with col2:
        st.markdown("""
        **Update Frequency:**
        - Annual updates for most datasets
        - Real-time API access
        - Historical data back to 1961
        - Provisional data flags
        """)

def render_technical_info():
    """Render technical information about the application."""
    
    st.markdown("## ⚙️ Technical Information")
    
    # Architecture overview
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
        **Technology Stack:**
        
        - **Frontend:** Streamlit (Python web framework)
        - **Data Processing:** Pandas, NumPy
        - **Visualizations:** Matplotlib, Seaborn, Plotly
        - **AI Integration:** OpenAI GPT-4/GPT-3.5
        - **Document Generation:** ReportLab (PDF), python-docx (Word)
        - **Data Source:** FAOSTAT REST API
        
        **Key Components:**
        
        - **Services Layer:** FAOSTAT API integration, AI processing, document generation
        - **Utils Layer:** Data processing, visualization generation, prompt templates
        - **Models Layer:** Data structures and type definitions
        - **Components Layer:** Reusable UI components
        - **Features Layer:** Advanced functionality modules
        """)
    
    # API information
    with st.expander("🔗 API Integration"):
        st.markdown("""
        **FAOSTAT API:**
        
        - Base URL: https://bulks-faostat.fao.org/production/
        - Data Format: CSV downloads via REST API
        - Caching: Intelligent caching for performance
        - Rate Limiting: Respectful API usage patterns
        
        **OpenAI API:**
        
        - Models: GPT-4, GPT-o3
        - Token Management: Efficient prompt design
        - Cost Optimization: Response caching
        - Error Handling: Graceful degradation
        """)
    
    # Performance specifications
    with st.expander("📈 Performance Specifications"):
        st.markdown("""
        **System Requirements:**
        
        - **Memory:** 4GB RAM minimum, 8GB recommended
        - **Storage:** 1GB available space
        - **Network:** Stable internet connection for API access
        - **Browser:** Modern web browser (Chrome, Firefox, Safari, Edge)
        
        **Performance Metrics:**
        
        - Dataset loading: 10-60 seconds (depending on size)
        - Visualization generation: 5-15 seconds
        - AI insight generation: 10-30 seconds
        - Report generation: 15-45 seconds
        """)

def render_usage_guide():
    """Render usage guide and best practices."""
    
    st.markdown("## 📖 Usage Guide")
    
    # Getting started
    with st.expander("🚀 Getting Started"):
        st.markdown("""
        **Step 1: Configuration**
        1. Add your OpenAI API key in the sidebar (for AI features)
        2. Configure output settings as needed
        3. Check system status indicators
        
        **Step 2: Explore Data**
        1. Browse available datasets in the Dataset Browser
        2. Preview dataset contents and metadata
        3. Select a dataset for analysis
        
        **Step 3: Analyze**
        1. Configure analysis filters (time, geography, items)
        2. Generate visualizations
        3. Create AI-powered insights
        4. Export results and reports
        
        **Step 4: Advanced Features**
        1. Try thematic analysis templates
        2. Use natural language queries
        3. Compare multiple datasets
        4. Save bookmarks for future reference
        """)
    
    # Best practices
    with st.expander("✅ Best Practices"):
        st.markdown("""
        **Data Analysis:**
        - Start with recent years (last 10) for clearer trends
        - Limit geographic scope for better visualization readability
        - Use appropriate aggregation methods for your research question
        - Check data quality flags and metadata
        
        **AI Queries:**
        - Be specific in your questions
        - Provide context about what you're looking for
        - Use follow-up questions to drill down into insights
        - Verify AI responses with the underlying data
        
        **Report Generation:**
        - Review all visualizations before generating reports
        - Customize report titles and author information
        - Include explanatory notes about methodology
        - Save multiple versions for different audiences
        
        **Performance Optimization:**
        - Use dataset previews to understand structure first
        - Apply filters early to reduce processing time
        - Cache expensive operations when possible
        - Monitor API usage and costs
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        **Data Loading Problems:**
        - Check internet connection
        - Verify FAOSTAT API availability
        - Try refreshing dataset list
        - Use smaller date ranges for large datasets
        
        **AI Features Not Working:**
        - Verify OpenAI API key is correctly configured
        - Check API credit balance
        - Try shorter, simpler queries first
        - Restart application if needed
        
        **Visualization Issues:**
        - Ensure data has appropriate structure
        - Check for missing values or data quality issues
        - Try different chart types
        - Reduce data complexity if charts are cluttered
        
        **Report Generation Failures:**
        - Verify all required dependencies are installed
        - Check available disk space
        - Ensure visualizations were generated successfully
        - Try generating with fewer visualizations
        """)

def render_credits():
    """Render credits and attribution."""
    
    st.markdown("## 🙏 Credits & Attribution")
    
    # Data attribution
    st.markdown("### 📊 Data Sources")
    st.markdown("""
    **FAOSTAT Database**
    - Food and Agriculture Organization of the United Nations (FAO)
    - License: [CC BY-NC-SA 3.0 IGO](https://creativecommons.org/licenses/by-nc-sa/3.0/igo/)
    - Website: [http://www.fao.org/faostat](http://www.fao.org/faostat)
    
    *"This application uses data from FAOSTAT, the statistical database of the Food and Agriculture 
    Organization of the United Nations. The data is provided under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO license."*
    """)
    
    # Technology credits
    st.markdown("### 🛠️ Technology")
    
    tech_credits = {
        "Core Framework": [
            "Streamlit - Web application framework",
            "Python - Programming language",
            "Pandas - Data manipulation and analysis",
            "NumPy - Numerical computing"
        ],
        "AI & Machine Learning": [
            "OpenAI - GPT models for natural language processing"
        ],
        "Visualization": [
            "Matplotlib - Statistical plotting",
            "Seaborn - Statistical data visualization", 
            "Plotly - Interactive visualizations"
        ],
        "Document Generation": [
            "ReportLab - PDF generation",
            "python-docx - Word document creation",
            "WeasyPrint - HTML to PDF conversion"
        ]
    }
    
    cols = st.columns(2)
    
    for i, (category, tools) in enumerate(tech_credits.items()):
        with cols[i % 2]:
            st.markdown(f"**{category}:**")
            for tool in tools:
                st.markdown(f"• {tool}")
    
    # Development and maintenance
    st.markdown("### 👨‍💻 Development")
    st.markdown("""
    **FAOSTAT Analytics** was developed to support agricultural research and policy analysis by providing 
    easy access to FAO data with AI-powered insights.
    
    **Version Information:**
    - Version: 1.0.0
    - Last Updated: December 2024
    - License: MIT License (application code)
    - Repository: [GitHub](https://github.com/your-org/faostat-analytics)
    """)
    
    # Disclaimer
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    **Important Notes:**
    
    - This application is an independent tool and is not officially affiliated with FAO
    - AI-generated insights should be verified against source data
    - Data accuracy depends on FAOSTAT database quality and timeliness
    - Users are responsible for appropriate interpretation and use of results
    - For official FAO publications, please refer to [FAO's official website](http://www.fao.org)
    
    **Data Usage:**
    - Respect FAOSTAT's data attribution requirements
    - Follow CC BY-NC-SA 3.0 IGO license terms
    - Cite data sources appropriately in publications
    - Check for updates and revisions of data
    """)
    
    # Contact and feedback
    st.markdown("### 📧 Contact & Feedback")
    st.markdown("""
    **Get Involved:**
    - Report issues: [GitHub Issues](https://github.com/your-org/faostat-analytics/issues)
    - Suggest features: [Feature Requests](https://github.com/your-org/faostat-analytics/discussions)
    - Contribute: [Contributing Guide](https://github.com/your-org/faostat-analytics/blob/main/CONTRIBUTING.md)
    - Documentation: [User Manual](https://faostat-analytics.readthedocs.io)
    
    **Support:**
    - Email: support@faostat-analytics.org
    - Community Forum: [Discussions](https://github.com/your-org/faostat-analytics/discussions)
    - Documentation: [Docs](https://docs.faostat-analytics.org)
    """)
    
    # Footer with build info
    st.markdown("---")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Page generated on {current_time} | FAOSTAT Analytics v1.0.0")