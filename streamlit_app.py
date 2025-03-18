import streamlit as st
import asyncio
import os
import time
import re
import pandas as pd
import logging
import tempfile
import base64
from pathlib import Path
from enhanced_report import fetch_competitor_data, format_excel_output, generate_pdf_report

# Configure logging to capture INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a custom StreamHandler that will send logs to Streamlit
class StreamlitHandler(logging.Handler):
    def __init__(self, st_container):
        super().__init__()
        self.st_container = st_container
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        # Update the logs in the Streamlit container
        log_text = "\n".join(self.logs)
        self.st_container.code(log_text)

# Function to create a download link for files
def get_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'

# Function to display PDF and Excel files
def display_file_preview(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        # Display PDF preview
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    elif file_ext == '.xlsx':
        # Display Excel preview
        df_dict = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in df_dict.items():
            if not df.empty:
                st.write(f"**Sheet: {sheet_name}**")
                st.dataframe(df.head(10))
                if len(df) > 10:
                    st.write(f"*Showing 10 of {len(df)} rows*")
                st.write("---")

# Async function to run the competitive analysis
async def run_analysis(company_name, competitors, log_container):
    # Create a list to hold competitor data
    all_competitor_data = []
    
    # Set up progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each competitor
    for i, competitor in enumerate(competitors):
        status_text.text(f"Analyzing {competitor} ({i+1}/{len(competitors)})")
        try:
            # Fetch data for this competitor
            data = await fetch_competitor_data(competitor)
            all_competitor_data.append(data)
        except Exception as e:
            logger.error(f"Failed to analyze {competitor}: {str(e)}")
            error_data = {
                "Competitor": competitor,
                "Company Overview": f"Error: {str(e)}",
                "Product Line": "Error: Analysis failed",
                "Features": "Error: Analysis failed",
                "Pricing": "Error: Analysis failed",
                "Customer Reviews & Segments": "Error: Analysis failed",
                "SWOT Analysis": "Error: Analysis failed",
                "Market Position": "Error: Analysis failed",
            }
            all_competitor_data.append(error_data)
        
        # Update progress
        progress_bar.progress((i + 1) / len(competitors))
    
    # Generate reports
    status_text.text("Generating Excel report...")
    excel_filename = format_excel_output(all_competitor_data, company_name)
    
    status_text.text("Generating PDF report...")
    pdf_filename = generate_pdf_report(all_competitor_data, company_name, excel_filename)
    
    status_text.text("Analysis complete!")
    progress_bar.progress(1.0)
    
    return excel_filename, pdf_filename

# Main function for Streamlit app
def main():
    st.set_page_config(
        page_title="AI Agentic Competitor Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #3B82F6;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .success-box {
            background-color: #ECFDF5;
            border-left: 5px solid #10B981;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">AI Agentic Competitor Analysis</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Generate comprehensive competitor analysis reports with just a few clicks. Our AI-powered tool will research and analyze your competitors, creating detailed Excel and PDF reports.</div>', unsafe_allow_html=True)
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown('<p class="sub-header">How it works</p>', unsafe_allow_html=True)
        st.write("1. Enter your company name")
        st.write("2. Enter competitor names (separated by commas)")
        st.write("3. Click 'Generate Analysis'")
        st.write("4. Wait for the analysis to complete")
        st.write("5. Download the reports")
        
        st.markdown("---")
        st.markdown('<p class="sub-header">About</p>', unsafe_allow_html=True)
        st.write("This tool uses OpenAI's Agents SDK to research and analyze competitors in real-time.")
        st.write("The analysis includes:")
        st.write("- Company Overview")
        st.write("- Product Lines")
        st.write("- Features")
        st.write("- Pricing")
        st.write("- Customer Reviews")
        st.write("- SWOT Analysis")
        st.write("- Market Position")
    
    # Main content area with form
    with st.form("competitor_form"):
        company_name = st.text_input("Your Company Name", placeholder="Enter your company name")
        competitor_input = st.text_input("Competitor Names (comma-separated)", 
                                       placeholder="Enter competitor names separated by commas")
        
        # Advanced options in an expander
        with st.expander("Advanced Options"):
            st.info("These options are set to optimal defaults and typically don't need adjustment.")
            max_competitors = st.slider("Maximum number of competitors to analyze", 1, 10, 5)
            st.write("**Note:** Each competitor analysis takes approximately 2-4 minutes.")
        
        # Submit button
        submitted = st.form_submit_button("Generate Analysis", type="primary")
    
    # Process form submission
    if submitted:
        if not company_name.strip():
            st.error("Please enter your company name")
        elif not competitor_input.strip():
            st.error("Please enter at least one competitor name")
        else:
            # Parse competitor names
            competitors = [comp.strip() for comp in competitor_input.split(",") if comp.strip()]
            
            # Limit number of competitors
            if len(competitors) > max_competitors:
                st.warning(f"Limiting analysis to {max_competitors} competitors")
                competitors = competitors[:max_competitors]
            
            # Display competitors being analyzed
            st.write("### Analysis Details")
            st.write(f"**Company:** {company_name}")
            st.write(f"**Competitors:** {', '.join(competitors)}")
            
            # Create container for logs
            st.write("### Analysis Progress")
            log_container = st.empty()
            
            # Add StreamlitHandler to root logger
            streamlit_handler = StreamlitHandler(log_container)
            root_logger = logging.getLogger()
            root_logger.addHandler(streamlit_handler)
            
            # Run the analysis
            with st.spinner("Analyzing competitors... (This may take several minutes)"):
                # Run asyncio event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                excel_file, pdf_file = loop.run_until_complete(run_analysis(company_name, competitors, log_container))
                loop.close()
            
            # Remove the StreamlitHandler
            root_logger.removeHandler(streamlit_handler)
            
            # Display success message
            st.markdown('<div class="success-box"><h3>Analysis Complete! 🎉</h3><p>Your competitor analysis reports are ready. You can preview and download them below.</p></div>', unsafe_allow_html=True)
            
            # Create tabs for the reports
            tab1, tab2 = st.tabs(["PDF Report", "Excel Report"])
            
            with tab1:
                st.write("### PDF Report Preview")
                display_file_preview(pdf_file)
                st.markdown(get_download_link(pdf_file, "📥 Download PDF Report"), unsafe_allow_html=True)
            
            with tab2:
                st.write("### Excel Report Preview")
                display_file_preview(excel_file)
                st.markdown(get_download_link(excel_file, "📥 Download Excel Report"), unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main() 