import asyncio
import logging
import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from agents import Agent, Runner, WebSearchTool
import time
import traceback
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# -----------------------------------------------------------------------------
# Load environment variables from .env file
# -----------------------------------------------------------------------------
load_dotenv()  # Loads variables from .env into os.environ
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Set up API configuration
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))  # 2 minutes default timeout
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # 3 retries by default
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # 5 seconds delay between retries

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Input Validation Models
# -----------------------------------------------------------------------------
class CompetitorInput(BaseModel):
    name: str = Field(..., min_length=1, description="Competitor company name")
    
    @validator('name')
    def name_must_not_be_blank(cls, v):
        if v.strip() == '':
            raise ValueError('Competitor name cannot be blank')
        return v

class UserInputs(BaseModel):
    company_name: str = Field(..., min_length=1, description="Your company name")
    competitors: List[str] = Field(..., min_items=1, description="List of competitor names")
    
    @validator('competitors')
    def competitors_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('At least one competitor must be provided')
        return v
    
    @validator('company_name')
    def company_name_must_not_be_blank(cls, v):
        if v.strip() == '':
            raise ValueError('Company name cannot be blank')
        return v

# -----------------------------------------------------------------------------
# Common instructions and formatting
# -----------------------------------------------------------------------------
CITATION_INSTRUCTIONS = """
Always include full source links and citations for all information.
Format citations as [Source: URL] at the end of each major section.
"""

FORMATTING_INSTRUCTIONS = """
Structure your output with clear headings and bullet points for readability.
Use markdown formatting for emphasis where appropriate.
Include a brief summary at the beginning and detailed analysis in subsequent sections.
"""

DATA_STRUCTURE_INSTRUCTIONS = """
For comparability and reporting, please structure your response with these specific data points:
1. SUMMARY: Provide a brief 2-3 sentence overview
2. MAIN_TEXT: Your full analysis goes here
3. KEY_METRICS: Include 3-5 key numerical metrics in this format:
   - metric_name1: value1
   - metric_name2: value2
4. COMPARISON_TABLE: Provide a comparison table in markdown format with at least 3 rows
5. CITATIONS: List all sources used
"""

# -----------------------------------------------------------------------------
# Define Specialized Agents for Each Data Section for a Competitor
# -----------------------------------------------------------------------------

# 1. Competitor Overview Agent
competitor_overview_agent = Agent(
    name="Competitor Overview Agent",
    instructions=(
        "You are a seasoned market researcher with over 15 years of experience. "
        "For the competitor provided in the input, produce a detailed company overview including company history, "
        "founding year, headquarters, leadership profiles, financial status, investments, revenue, key milestones, and strategic positioning. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- founded_year: numeric year (e.g., 1998)\n"
        "- revenue_usd_millions: latest annual revenue in USD millions (e.g., 4500)\n"
        "- employee_count: number of employees (e.g., 12000)\n"
        "- market_cap_usd_billions: market capitalization in USD billions for public companies (e.g., 85.3)\n"
        "- growth_rate_percent: latest annual growth rate percentage (e.g., 14.2)\n"
    ),
    tools=[WebSearchTool()],
)

# 2. Product Line Agent
product_line_agent = Agent(
    name="Product Line Agent",
    instructions=(
        "You are an expert in product analysis. For the competitor provided in the input, provide a comprehensive description of their product line. "
        "Detail each product or service, its features, unique selling points, and any performance or market data. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- product_count: total number of distinct products/services (e.g., 8)\n"
        "- flagship_product_revenue_percent: percentage of revenue from flagship product (e.g., 45)\n"
        "- product_categories_count: number of product categories (e.g., 3)\n"
        "\nFor the COMPARISON_TABLE, create a table comparing the top 3-5 products with columns for name, category, target market, and key strength."
    ),
    tools=[WebSearchTool()],
)

# 3. Features Agent
features_agent = Agent(
    name="Features Comparison Agent",
    instructions=(
        "You are an expert in product features. For the competitor provided in the input, provide a detailed analysis of the product features. "
        "Include technical specifications, user experience insights, integration capabilities, and competitive advantages. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- unique_features_count: number of unique features (e.g., 12)\n"
        "- average_feature_rating: average rating of features out of 10 (e.g., 8.4)\n"
        "- integration_partners_count: number of integration partners (e.g., 35)\n"
        "\nFor the COMPARISON_TABLE, create a features comparison table with rows for each key feature category and a rating out of 10."
    ),
    tools=[WebSearchTool()],
)

# 4. Pricing Agent
pricing_agent = Agent(
    name="Pricing Agent",
    instructions=(
        "You are a pricing analyst. For the competitor provided in the input, produce a detailed analysis of their pricing models. "
        "Include subscription models, one-time fees, discount structures, and any industry benchmark comparisons. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- lowest_price_usd: lowest price point in USD (e.g., 9.99)\n"
        "- highest_price_usd: highest price point in USD (e.g., 499)\n"
        "- average_customer_spend_usd: average customer spend in USD (e.g., 125)\n"
        "- pricing_tiers_count: number of pricing tiers (e.g., 4)\n"
        "\nFor the COMPARISON_TABLE, create a pricing tier comparison with columns for tier name, price, key features, and target customer."
    ),
    tools=[WebSearchTool()],
)

# 5. Customer Reviews & Segments Agent
customer_reviews_agent = Agent(
    name="Customer Reviews Agent",
    instructions=(
        "You are an expert in customer insights. For the competitor provided in the input, gather detailed data on customer segments and reviews. "
        "Include quantitative ratings, qualitative feedback, trends in customer satisfaction, and any notable commentary. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- average_rating: average customer rating out of 5 (e.g., 4.2)\n"
        "- nps_score: Net Promoter Score if available (e.g., 42)\n"
        "- review_count: total number of reviews found (e.g., 1248)\n"
        "- positive_review_percent: percentage of positive reviews (e.g., 78)\n"
        "\nFor the COMPARISON_TABLE, create a table showing ratings across different platforms/review sites with columns for platform, average rating, and review count."
    ),
    tools=[WebSearchTool()],
)

# 6. SWOT Analysis Agent
swot_agent = Agent(
    name="SWOT Analysis Agent",
    instructions=(
        "You are a strategic analyst with deep expertise. For the competitor provided in the input, provide a comprehensive SWOT analysis. "
        "Detail the strengths, weaknesses, opportunities, and threats using specific metrics, qualitative insights, and industry trends. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- strengths_count: number of key strengths identified (e.g., 7)\n"
        "- weaknesses_count: number of key weaknesses identified (e.g., 5)\n"
        "- opportunities_count: number of key opportunities identified (e.g., 6)\n"
        "- threats_count: number of key threats identified (e.g., 4)\n"
        "\nFor the COMPARISON_TABLE, create a prioritized SWOT table ranking each item by importance score out of 10."
    ),
    tools=[WebSearchTool()],
)

# 7. Added Market Position Agent
market_position_agent = Agent(
    name="Market Position Agent",
    instructions=(
        "You are a market analysis expert with deep knowledge of competitive positioning. "
        "For the competitor provided in the input, analyze their market position, market share, competitive landscape, "
        "positioning strategy, and growth trajectory in their key markets. "
        f"{CITATION_INSTRUCTIONS} {FORMATTING_INSTRUCTIONS} {DATA_STRUCTURE_INSTRUCTIONS}\n\n"
        "For the KEY_METRICS section, include these specific metrics where available:\n"
        "- global_market_share_percent: global market share percentage (e.g., 12.5)\n" 
        "- market_position_rank: rank in their primary market (e.g., 2)\n"
        "- yoy_market_share_change: year-over-year market share change in percentage points (e.g., 1.8)\n"
        "\nFor the COMPARISON_TABLE, create a market share comparison table for top 5 competitors in their main market."
    ),
    tools=[WebSearchTool()],
)

# -----------------------------------------------------------------------------
# Data Extraction and Processing Functions
# -----------------------------------------------------------------------------
def extract_structured_data(text):
    """
    Extract structured data from agent outputs in the predefined format.
    
    Args:
        text: The text output from an agent
        
    Returns:
        Dictionary with extracted data sections
    """
    data = {
        "SUMMARY": "",
        "MAIN_TEXT": "",
        "KEY_METRICS": {},
        "COMPARISON_TABLE": "",
        "CITATIONS": []
    }
    
    # Extract sections using markers
    sections = {
        "SUMMARY": ["SUMMARY:", "MAIN_TEXT:"],
        "MAIN_TEXT": ["MAIN_TEXT:", "KEY_METRICS:"],
        "KEY_METRICS": ["KEY_METRICS:", "COMPARISON_TABLE:"],
        "COMPARISON_TABLE": ["COMPARISON_TABLE:", "CITATIONS:"],
        "CITATIONS": ["CITATIONS:", None]
    }
    
    # Handle case where the structure isn't followed
    if not all(marker in text for marker in ["SUMMARY:", "MAIN_TEXT:", "KEY_METRICS:"]):
        # If the structure isn't followed, just return the full text as MAIN_TEXT
        data["MAIN_TEXT"] = text
        return data
    
    # Extract each section
    for section, (start_marker, end_marker) in sections.items():
        if start_marker in text:
            start_idx = text.find(start_marker) + len(start_marker)
            if end_marker and end_marker in text:
                end_idx = text.find(end_marker)
                section_text = text[start_idx:end_idx].strip()
            else:
                section_text = text[start_idx:].strip()
                
            if section == "KEY_METRICS":
                # Parse metrics
                metrics = {}
                for line in section_text.split('\n'):
                    line = line.strip()
                    if line.startswith('-') and ':' in line:
                        # Format: - metric_name: value
                        line = line[1:].strip()  # Remove the dash
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key, value = parts[0].strip(), parts[1].strip()
                            # Try to convert numeric values
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except (ValueError, TypeError):
                                pass
                            metrics[key] = value
                data[section] = metrics
            else:
                data[section] = section_text
    
    return data

# -----------------------------------------------------------------------------
# Helper Functions for Retries and Error Handling
# -----------------------------------------------------------------------------
async def run_with_retry(agent, query, section_name, competitor, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """
    Run an agent with proper retry logic, creating a new coroutine each time.
    
    Args:
        agent: The agent to run
        query: The query to send to the agent
        section_name: Name of the section (for logging)
        competitor: Name of the competitor being analyzed
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Agent result or error message
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Important: Create a new coroutine on each attempt
            result = await asyncio.wait_for(
                Runner.run(agent, input=query),
                timeout=TIMEOUT_SECONDS
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt} for {section_name} timed out after {TIMEOUT_SECONDS}s.")
            if attempt < max_retries:
                logger.warning(f"Retrying {section_name} in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts for {section_name} timed out.")
                return f"Error: Data collection timed out after {max_retries} attempts."
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Attempt {attempt} failed for {section_name}: {error_msg}. Retrying in {retry_delay} seconds...")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Error fetching {section_name} for {competitor}: {error_msg}")
                return f"Error: {error_msg}"

# -----------------------------------------------------------------------------
# Function to Run All Agents for a Given Competitor
# -----------------------------------------------------------------------------
async def fetch_competitor_data(competitor: str) -> dict:
    """
    For a given competitor name, run all specialized agents to fetch detailed data.
    Uses proper retry logic and error handling.
    """
    logger.info(f"Fetching data for competitor: {competitor}")
    
    # Validate competitor input
    validated_input = CompetitorInput(name=competitor)
    competitor = validated_input.name
    
    # Prepare the query
    query = f"Research comprehensive information about {competitor} as a company and its products/services."
    
    # Dictionary to hold all section results
    results = {
        "Competitor": competitor,
    }
    
    # Define all sections and their corresponding agents
    sections = {
        "Company Overview": competitor_overview_agent,
        "Product Line": product_line_agent,
        "Features": features_agent,
        "Pricing": pricing_agent,
        "Customer Reviews & Segments": customer_reviews_agent,
        "SWOT Analysis": swot_agent,
        "Market Position": market_position_agent
    }
    
    # Run each section sequentially with proper retry logic
    # This avoids overwhelming the API and is more reliable than concurrent execution
    for section_name, agent in sections.items():
        logger.info(f"Starting {section_name} analysis for {competitor}")
        
        # Run the agent with retry logic
        result = await run_with_retry(agent, query, section_name, competitor)
        
        # Store the result
        if isinstance(result, str):
            # This is an error message string
            results[section_name] = result
            results[f"{section_name}_Structured"] = {"MAIN_TEXT": result}
        else:
            # This is a successful agent result
            text_output = result.final_output
            results[section_name] = text_output
            
            # Extract and store structured data
            structured_data = extract_structured_data(text_output)
            results[f"{section_name}_Structured"] = structured_data
            
            # Store usage data if available
            if hasattr(result, 'usage') and result.usage:
                if "Metadata" not in results:
                    results["Metadata"] = {}
                results["Metadata"][f"{section_name}_Usage"] = result.usage
        
        logger.info(f"Completed {section_name} analysis for {competitor}")
    
    return results

# -----------------------------------------------------------------------------
# Progress Display
# -----------------------------------------------------------------------------
class ProgressTracker:
    """Track and display progress for multi-competitor analysis."""
    
    def __init__(self, total_competitors, competitors):
        self.total = total_competitors
        self.completed = 0
        self.start_time = time.time()
        self.competitors = competitors
        self.current_competitor = None

    def start_competitor(self, competitor):
        """Mark that analysis of a competitor has started."""
        self.current_competitor = competitor
        self._update_display()
    
    def complete_competitor(self, competitor):
        """Mark that analysis of a competitor is complete."""
        self.completed += 1
        self._update_display()
    
    def _update_display(self):
        """Update the progress display."""
        elapsed = time.time() - self.start_time
        percent = self.completed / self.total * 100 if self.total > 0 else 0
        
        # Calculate ETA
        if self.completed > 0:
            avg_time_per_competitor = elapsed / self.completed
            eta = avg_time_per_competitor * (self.total - self.completed)
            eta_minutes = int(eta // 60)
            eta_seconds = int(eta % 60)
            time_str = f"{eta_minutes}m {eta_seconds}s"
        else:
            time_str = "calculating..."
        
        # Current status
        if self.current_competitor:
            status = f"Analyzing {self.current_competitor} ({self.completed + 1}/{self.total})"
        else:
            status = f"Completed {self.completed}/{self.total}"
        
        # Progress bar (30 chars wide)
        progress_chars = 30
        filled = int(percent / 100 * progress_chars)
        bar = "█" * filled + "░" * (progress_chars - filled)
        
        # Format and print the progress line
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        progress_line = f"{status} | [{bar}] {percent:.1f}% | Elapsed: {elapsed_min}m {elapsed_sec}s | ETA: {time_str}"
        
        # Clear line and write new progress
        print(f"\r{progress_line}", end="")
    
    def complete(self):
        """Mark the entire process as complete."""
        self.completed = self.total
        elapsed = time.time() - self.start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        print(f"\nAnalysis complete! Total time: {elapsed_min}m {elapsed_sec}s")

# -----------------------------------------------------------------------------
# Function to Gather User Inputs
# -----------------------------------------------------------------------------
def gather_user_inputs() -> UserInputs:
    """
    Collect and validate user inputs for company name and competitors.
    """
    print("\n" + "="*80)
    print("Welcome to the Competitor Analysis Excel Generator!")
    print("="*80 + "\n")
    
    while True:
        try:
            company_name = input("Enter your company name: ").strip()
            competitor_names = input("Enter competitor names (comma-separated): ").strip()
            
            competitors = [comp.strip() for comp in competitor_names.split(",") if comp.strip()]
            
            # Validate inputs using Pydantic model
            inputs = UserInputs(
                company_name=company_name,
                competitors=competitors
            )
            
            # Ask for confirmation
            print("\nYou've entered:")
            print(f"Your company: {inputs.company_name}")
            print(f"Competitors to analyze: {', '.join(inputs.competitors)}")
            print(f"\nTimeout set to: {TIMEOUT_SECONDS} seconds per API call")
            print(f"Max retries: {MAX_RETRIES} times")
            confirm = input("\nIs this correct? (y/n): ").lower().strip()
            
            if confirm == 'y':
                return inputs
            else:
                print("Let's try again.\n")
        
        except Exception as e:
            print(f"\nError in input: {str(e)}")
            print("Please try again.\n")

# -----------------------------------------------------------------------------
# Format text for better readability by removing markdown symbols and paragraph numbers
# -----------------------------------------------------------------------------
def clean_text_for_display(text):
    """
    Remove markdown formatting and clean up text for better readability in reports.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text without markdown symbols or unnecessary formatting
    """
    # Replace markdown bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove ** bold markers
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove * italic markers
    
    # Remove any backticks for code
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Clean up any HTML tags that might have been included
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove any other markdown formatting indicators
    text = re.sub(r'_{2,}(.*?)_{2,}', r'\1', text)  # Remove __ underscores
    
    # Fix spacing issues
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize multiple newlines
    
    return text.strip()

# -----------------------------------------------------------------------------
# Modified section for Excel formatting to properly use the SWOT table function
# -----------------------------------------------------------------------------
def format_excel_output(all_competitor_data, company_name):
    """
    Create a comprehensive, multi-sheet Excel file with detailed competitor analysis,
    with each point/paragraph in a separate row for better readability.
    
    Args:
        all_competitor_data: List of competitor data dictionaries
        company_name: User's company name
        
    Returns:
        Filename of the generated Excel file
    """
    # Create Excel file name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_company_name = ''.join(c if c.isalnum() else '_' for c in company_name)
    filename = f"{safe_company_name}_competitor_analysis_{timestamp}.xlsx"
    
    # Create Excel writer with xlsxwriter engine for better formatting
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Create formats
        title_format = workbook.add_format({
            'bold': True, 'font_size': 14, 'align': 'center', 
            'valign': 'vcenter', 'bg_color': '#4472C4', 'font_color': 'white'
        })
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 
            'text_wrap': True, 'valign': 'top'
        })
        subheader_format = workbook.add_format({
            'bold': True, 'bg_color': '#E2EFDA', 'border': 1,
            'text_wrap': True, 'valign': 'top'
        })
        text_format = workbook.add_format({
            'text_wrap': True, 'valign': 'top', 'border': 1
        })
        point_format = workbook.add_format({
            'text_wrap': True, 'valign': 'top', 'border': 1,
            'indent': 1  # Add indentation for bullet points
        })
        
        # Format settings for SWOT table
        format_settings = {
            'workbook': workbook,
            'title_format': title_format,
            'header_format': header_format,
            'subheader_format': subheader_format,
            'text_format': text_format,
            'point_format': point_format
        }
        
        # -----------------------------------------------------------------------
        # Create a dedicated SWOT Analysis worksheet 
        # -----------------------------------------------------------------------
        swot_worksheet = workbook.add_worksheet('SWOT Analysis')
        swot_worksheet.set_column('A:C', 30)  # Set column width for strengths/opportunities
        swot_worksheet.set_column('D:F', 30)  # Set column width for weaknesses/threats
        
        # Add title
        swot_worksheet.merge_range('A1:F1', f'SWOT Analysis for Competitors', title_format)
        
        # Initialize row for SWOT tables
        swot_row = 2
        
        # Add a SWOT table for each competitor
        for data in all_competitor_data:
            competitor_name = data['Competitor']
            
            # Check if SWOT data exists
            if "SWOT Analysis_Structured" in data and "MAIN_TEXT" in data["SWOT Analysis_Structured"]:
                # Add a SWOT table using our helper function
                swot_row = create_swot_table(swot_worksheet, data, swot_row, competitor_name, format_settings)
            else:
                # Just add a header if no SWOT data
                swot_worksheet.merge_range(swot_row, 0, swot_row, 5, f"SWOT Analysis: {competitor_name} - No data available", title_format)
                swot_row += 2
        
        # -----------------------------------------------------------------------
        # Create Detailed Analysis Sheets for Each Section with improved formatting
        # -----------------------------------------------------------------------
        sections = [
            'Company Overview', 
            'Product Line', 
            'Features', 
            'Pricing', 
            'Customer Reviews & Segments', 
            'SWOT Analysis', 
            'Market Position'
        ]
        
        for section in sections:
            # Create a worksheet for this section
            sheet_name = f"{section} Details"[:31]
            worksheet = workbook.add_worksheet(sheet_name)
            
            # Add title
            worksheet.merge_range('A1:C1', f'{section} Detailed Analysis', title_format)
            
            # Add headers
            worksheet.write(1, 0, "Competitor", header_format)
            worksheet.write(1, 1, "Section", header_format)
            worksheet.write(1, 2, "Content", header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 20)  # Competitor column
            worksheet.set_column('B:B', 25)  # Section column
            worksheet.set_column('C:Z', 30)  # Content and table columns - extend to column Z for tables
            
            # Start row for data
            row = 2
            
            # Process each competitor's data
            for data in all_competitor_data:
                competitor_name = data['Competitor']
                
                # Get the text data for this section
                structured_key = f"{section}_Structured"
                if structured_key in data and 'MAIN_TEXT' in data[structured_key]:
                    main_text = data[structured_key]['MAIN_TEXT']
                else:
                    main_text = data.get(section, "Data not available")
                
                # Skip if no data
                if main_text.strip() == "Data not available" or not main_text.strip():
                    worksheet.write(row, 0, competitor_name, text_format)
                    worksheet.write(row, 1, "No Data", text_format)
                    worksheet.write(row, 2, "No data available for this section", text_format)
                    row += 1
                    continue
                
                # Get the summary
                summary = ""
                if structured_key in data and 'SUMMARY' in data[structured_key]:
                    summary = data[structured_key]['SUMMARY']
                    # Clean the summary text
                    summary = clean_text_for_display(summary)
                
                # Write the summary row if available
                if summary:
                    worksheet.write(row, 0, competitor_name, text_format)
                    worksheet.write(row, 1, "Summary", subheader_format)
                    worksheet.write(row, 2, summary, text_format)
                    row += 1
                
                # Split the text into paragraphs and sections
                # First look for markdown headers
                sections_dict = {}
                current_section = "General Information"
                
                # Try to split by markdown headers (## Header) first
                lines = main_text.split('\n')
                current_content = []
                
                for line in lines:
                    if re.match(r'^##?\s+', line):  # Matches # Header or ## Header
                        # If we have content for the previous section, save it
                        if current_content:
                            sections_dict[current_section] = '\n'.join(current_content).strip()
                            current_content = []
                        
                        # Start a new section
                        current_section = re.sub(r'^##?\s+', '', line).strip()
                    else:
                        current_content.append(line)
                
                # Save the last section
                if current_content:
                    sections_dict[current_section] = '\n'.join(current_content).strip()
                
                # If no headers were found, try to split by blank lines
                if len(sections_dict) <= 1 and len(main_text) > 200:
                    sections_dict = {}
                    paragraphs = re.split(r'\n\s*\n', main_text)
                    
                    for i, para in enumerate(paragraphs):
                        para = para.strip()
                        if para:
                            # Try to extract a title from the paragraph
                            first_line = para.split('\n')[0].strip()
                            if len(first_line) < 100 and not first_line.startswith('-') and not first_line.startswith('*'):
                                sections_dict[first_line] = '\n'.join(para.split('\n')[1:]).strip()
                            else:
                                # Don't use "Paragraph X" numbering - just use the content
                                sections_dict["Information"] = para
                
                # Write each section to the Excel sheet
                for section_title, content in sections_dict.items():
                    # Clean the content, removing markdown
                    content = clean_text_for_display(content)
                    
                    worksheet.write(row, 0, competitor_name, text_format)
                    worksheet.write(row, 1, section_title, subheader_format)
                    
                    # Check for table content - handle it differently
                    if '|' in content and '-|-' in content:
                        # This might be a markdown table - convert it to more readable form
                        worksheet.write(row, 2, "Table:", text_format)
                        row += 1
                        
                        # Try to parse and format the table
                        try:
                            table_rows = [line.strip() for line in content.split('\n') if line.strip() and '|' in line]
                            
                            # Skip the separator row (---|---|---)
                            header_row = table_rows[0] if table_rows else "| |"
                            data_rows = [r for r in table_rows[1:] if not re.match(r'^\s*[\-\|]+\s*$', r)]
                            
                            # Parse header columns
                            header_cols = [col.strip() for col in header_row.split('|')]
                            header_cols = [col for col in header_cols if col]  # Remove empty columns
                            
                            # If no header columns were found, create default
                            if not header_cols:
                                header_cols = ["Column 1", "Column 2"]
                            
                            # Write the header row
                            for col_idx, header in enumerate(header_cols):
                                worksheet.write(row, col_idx + 2, header, header_format)
                            row += 1
                            
                            # Parse and write data rows
                            for data_row in data_rows:
                                data_cols = [col.strip() for col in data_row.split('|')]
                                data_cols = [col for col in data_cols if col]  # Remove empty columns
                                
                                # If no data cols, skip this row
                                if not data_cols:
                                    continue
                                    
                                # Ensure we have enough columns
                                while len(data_cols) < len(header_cols):
                                    data_cols.append("")
                                
                                # Write each cell
                                for col_idx, cell in enumerate(data_cols):
                                    if col_idx < len(header_cols):  # Only write for defined columns
                                        worksheet.write(row, col_idx + 2, cell, text_format)
                                row += 1
                                
                            # Add an empty row after the table
                            row += 1
                        except Exception as e:
                            # If table parsing fails, just write the content as is
                            logger.warning(f"Failed to parse table: {str(e)}")
                            worksheet.write(row, 2, content, text_format)
                            row += 1
                    # Special handling for SWOT Analysis
                    elif section == 'SWOT Analysis' and any(keyword in section_title.lower() for keyword in ['swot', 'strength', 'weakness', 'opportunit', 'threat']):
                        # Try to identify if this is part of a SWOT analysis
                        swot_part = None
                        if 'strength' in section_title.lower():
                            swot_part = "Strengths"
                        elif 'weakness' in section_title.lower():
                            swot_part = "Weaknesses"
                        elif 'opportunit' in section_title.lower():
                            swot_part = "Opportunities"
                        elif 'threat' in section_title.lower():
                            swot_part = "Threats"
                        else:
                            swot_part = section_title
                            
                        worksheet.write(row, 2, f"{swot_part}:", text_format)
                        row += 1
                        
                        # Break the content into bullet points
                        bullet_points = re.findall(r'(?:^|\n)[*\-•]+(.*?)(?=(?:\n[*\-•]+|\n\n|$))', content, re.DOTALL)
                        
                        if bullet_points:
                            for point in bullet_points:
                                worksheet.write(row, 2, "• " + point.strip(), point_format)
                                row += 1
                        else:
                            # Split by sentences if no bullet points
                            sentences = re.split(r'(?<=[.!?])\s+', content)
                            for sentence in sentences:
                                if sentence.strip():
                                    worksheet.write(row, 2, "• " + sentence.strip(), point_format)
                                    row += 1
                    # If content is a long paragraph, break it down into bullet points 
                    # or smaller sections for better readability
                    elif len(content) > 300:
                        # Try to find bullet points
                        bullet_points = re.findall(r'(?:^|\n)[*\-•]+(.*?)(?=(?:\n[*\-•]+|\n\n|$))', content, re.DOTALL)
                        
                        if bullet_points:
                            # Write each bullet point as a separate entry
                            for i, point in enumerate(bullet_points):
                                if i == 0:
                                    # For the first point, use the same row
                                    worksheet.write(row, 2, "• " + point.strip(), point_format)
                                    row += 1
                                else:
                                    worksheet.write(row, 0, "", text_format)
                                    worksheet.write(row, 1, "", text_format)
                                    worksheet.write(row, 2, "• " + point.strip(), point_format)
                                    row += 1
                        else:
                            # If no bullet points, split by sentences for long content
                            sentences = re.split(r'(?<=[.!?])\s+', content)
                            current_text = ""
                            
                            for i, sentence in enumerate(sentences):
                                if i == 0 or len(current_text) + len(sentence) > 300:
                                    if i > 0:
                                        # Write the current batch and start a new one
                                        worksheet.write(row, 0, "", text_format)
                                        worksheet.write(row, 1, "", text_format)
                                        worksheet.write(row, 2, current_text, text_format)
                                        row += 1
                                    
                                    current_text = sentence
                                else:
                                    current_text += " " + sentence
                            
                            # Write the last batch
                            if current_text:
                                worksheet.write(row, 2, current_text, text_format)
                                row += 1
                    else:
                        # For shorter content, write it as is
                        worksheet.write(row, 2, content, text_format)
                        row += 1
                
                # Get the key metrics
                if structured_key in data and 'KEY_METRICS' in data[structured_key]:
                    metrics = data[structured_key]['KEY_METRICS']
                    if metrics:
                        worksheet.write(row, 0, competitor_name, text_format)
                        worksheet.write(row, 1, "Key Metrics", subheader_format)
                        
                        # Create a more structured table for metrics
                        worksheet.write(row, 2, "Metric", header_format)
                        worksheet.write(row, 3, "Value", header_format)
                        row += 1
                        
                        # Write each metric on its own row
                        for metric_name, metric_value in metrics.items():
                            # Format metric name for better display (capitalize, replace underscores)
                            formatted_name = metric_name.replace('_', ' ').title()
                            worksheet.write(row, 2, formatted_name, text_format)
                            worksheet.write(row, 3, metric_value, text_format)
                            row += 1
                
                # Add comparison table if available
                if structured_key in data and 'COMPARISON_TABLE' in data[structured_key]:
                    table_text = data[structured_key]['COMPARISON_TABLE']
                    if table_text:
                        worksheet.write(row, 0, competitor_name, text_format)
                        worksheet.write(row, 1, "Comparison Table", subheader_format)
                        
                        # Try to parse and format the markdown table
                        try:
                            table_rows = [line.strip() for line in table_text.split('\n') if line.strip() and '|' in line]
                            
                            if len(table_rows) >= 2:  # Need at least header and separator
                                # Skip the separator row (---|---|---)
                                header_row = table_rows[0] if table_rows else "| |"
                                data_rows = [r for r in table_rows[1:] if not re.match(r'^\s*[\-\|]+\s*$', r)]
                                
                                # Parse header columns
                                header_cols = [col.strip() for col in header_row.split('|')]
                                header_cols = [col for col in header_cols if col]  # Remove empty columns
                                
                                # If no header columns were found, create default
                                if not header_cols:
                                    header_cols = ["Column 1", "Column 2"]
                                
                                # Write the header row
                                for col_idx, header in enumerate(header_cols):
                                    worksheet.write(row, col_idx + 2, header, header_format)
                                row += 1
                                
                                # Parse and write data rows
                                for data_row in data_rows:
                                    data_cols = [col.strip() for col in data_row.split('|')]
                                    data_cols = [col for col in data_cols if col]  # Remove empty columns
                                    
                                    # If no data cols, skip this row
                                    if not data_cols:
                                        continue
                                    
                                    # Ensure we have enough columns
                                    while len(data_cols) < len(header_cols):
                                        data_cols.append("")
                                    
                                    # Write each cell
                                    for col_idx, cell in enumerate(data_cols):
                                        if col_idx < len(header_cols):  # Only write for defined columns
                                            worksheet.write(row, col_idx + 2, cell, text_format)
                                row += 1
                            else:
                                # Not enough rows for a proper table
                                worksheet.write(row, 2, clean_text_for_display(table_text), text_format)
                                row += 1
                        except Exception as e:
                            # If table parsing fails, just write the content as is
                            logger.warning(f"Failed to parse comparison table: {str(e)}")
                            worksheet.write(row, 2, clean_text_for_display(table_text), text_format)
                            row += 1
                
                # Add blank row between competitors
                worksheet.write(row, 0, "", text_format)
                worksheet.write(row, 1, "", text_format)
                worksheet.write(row, 2, "", text_format)
                row += 1
                
        # Add README sheet with instructions
        readme = workbook.add_worksheet('README')
        readme.set_column('A:A', 100)
        readme.write(0, 0, "HOW TO USE THIS COMPETITOR ANALYSIS REPORT", title_format)
        readme.write(1, 0, """
This report contains a detailed analysis of the specified competitors organized by topic:

Each sheet contains detailed information about one aspect of the competitors:
1. Company Overview Details
2. Product Line Details
3. Features Details
4. Pricing Details
5. Customer Reviews & Segments Details
6. SWOT Analysis Details and SWOT Analysis sheet
7. Market Position Details

Within each sheet, the information is organized with:
- Competitor name in Column A
- Section type in Column B (Summary, General Information, Key Metrics, etc.)
- Detailed content in Column C, broken down into readable chunks
- Tables are formatted with proper headers and data cells

The SWOT Analysis sheet contains a dedicated quadrant-based SWOT analysis for each competitor.

A PDF version of this report is also available with the same filename but .pdf extension.

This report was generated automatically using OpenAI's Agents SDK.
        """, text_format)
    
    return filename

# -----------------------------------------------------------------------------
# PDF Report Generation
# -----------------------------------------------------------------------------
def generate_pdf_report(all_competitor_data, company_name, excel_filename):
    """
    Generate a comprehensive PDF report from competitor data
    
    Args:
        all_competitor_data: List of competitor data dictionaries
        company_name: User's company name
        excel_filename: Name of the Excel file already generated (for reference)
        
    Returns:
        Filename of the generated PDF report
    """
    # Create PDF file name based on the Excel filename
    pdf_filename = excel_filename.replace('.xlsx', '.pdf')
    
    # Set up the document with better styling
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names to avoid conflicts
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=12,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=8,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6,
        spaceBefore=12,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=6,
        spaceAfter=6,
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceBefore=3,
        spaceAfter=3,
        bulletIndent=10,
        leading=14
    ))
    
    styles.add(ParagraphStyle(
        name='CustomTableHeader',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,  # center
        textColor=colors.white,
        backColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomTableCell',
        parent=styles['Normal'],
        fontSize=9,
        wordWrap=True,
        leading=12
    ))
    
    content = []
    
    # Cover Page
    content.append(Paragraph(f"Competitor Analysis Report", styles['CustomTitle']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Prepared for: {company_name}", styles['CustomHeading1']))
    content.append(Spacer(1, 36))
    
    # Add date
    current_date = time.strftime("%B %d, %Y")
    content.append(Paragraph(f"Report Date: {current_date}", styles['CustomNormal']))
    content.append(Spacer(1, 12))
    
    # List competitors
    content.append(Paragraph("Competitors Analyzed:", styles['CustomHeading2']))
    for data in all_competitor_data:
        competitor = data['Competitor']
        content.append(Paragraph(f"• {competitor}", styles['CustomBullet']))
    
    # Add page break after cover
    content.append(PageBreak())
    
    # Table of Contents
    content.append(Paragraph("Table of Contents", styles['CustomTitle']))
    content.append(Spacer(1, 12))
    
    # Add TOC entries
    toc_items = [
        "1. Executive Summary",
        "2. Competitor Profiles",
    ]
    
    # Add each section to TOC
    section_num = 3
    for section in ['Company Overview', 'Product Line', 'Features', 'Pricing', 
                  'Customer Reviews & Segments', 'SWOT Analysis', 
                  'Market Position', 'Technology & Innovation']:
        toc_items.append(f"{section_num}. {section} Analysis")
        section_num += 1
    
    # Add conclusion to TOC
    toc_items.append(f"{section_num}. Conclusion and Recommendations")
    
    # Format TOC
    for item in toc_items:
        content.append(Paragraph(item, styles['CustomNormal']))
        content.append(Spacer(1, 6))
    
    content.append(PageBreak())
    
    # Executive Summary
    content.append(Paragraph("1. Executive Summary", styles['CustomHeading1']))
    content.append(Spacer(1, 12))
    
    # Write a brief summary of the analysis
    summary_text = f"""
    This report provides a comprehensive competitive analysis of {len(all_competitor_data)} competitors 
    in relation to {company_name}. It covers company overviews, product offerings, features, 
    pricing strategies, customer reviews, SWOT analyses, and market positioning. The analysis
    is designed to help {company_name} understand the competitive landscape and identify
    strategic opportunities for growth and differentiation.
    """
    
    # Clean the text to remove markdown and extra formatting
    summary_text = clean_text_for_display(summary_text)
    content.append(Paragraph(summary_text, styles['CustomNormal']))
    content.append(Spacer(1, 12))
    
    # Key Findings
    content.append(Paragraph("Key Findings:", styles['CustomHeading2']))
    
    # Add some general findings about the competitors
    for i, data in enumerate(all_competitor_data, 1):
        competitor = data['Competitor']
        
        # Get a summary if available, otherwise create one
        if 'Company Overview_Structured' in data and 'SUMMARY' in data['Company Overview_Structured']:
            competitor_summary = data['Company Overview_Structured']['SUMMARY']
        else:
            competitor_summary = f"Analysis of {competitor}'s market position and offerings."
        
        # Clean the text
        competitor_summary = clean_text_for_display(competitor_summary)
        content.append(Paragraph(f"• {competitor}: {competitor_summary}", styles['CustomBullet']))
    
    content.append(Spacer(1, 12))
    content.append(PageBreak())
    
    # Competitor Profiles
    content.append(Paragraph("2. Competitor Profiles", styles['CustomHeading1']))
    
    # Create a summary profile for each competitor
    for data in all_competitor_data:
        competitor = data['Competitor']
        content.append(Paragraph(f"{competitor}", styles['CustomHeading2']))
        
        # Extract key information about the competitor if available
        overview = ""
        if 'Company Overview_Structured' in data and 'MAIN_TEXT' in data['Company Overview_Structured']:
            overview = data['Company Overview_Structured']['MAIN_TEXT']
        else:
            overview = data.get('Company Overview', f"No detailed information available for {competitor}.")
        
        # Clean text
        overview = clean_text_for_display(overview)
        content.append(Paragraph(overview[:500] + "...", styles['CustomNormal']))
        
        # Add key metrics if available
        if 'Company Overview_Structured' in data and 'KEY_METRICS' in data['Company Overview_Structured']:
            metrics = data['Company Overview_Structured']['KEY_METRICS']
            if metrics:
                content.append(Spacer(1, 6))
                content.append(Paragraph("Key Metrics:", styles['CustomHeading2']))
                
                # Create a table for metrics
                metric_data = []
                metric_data.append([Paragraph("Metric", styles['CustomTableHeader']), 
                                   Paragraph("Value", styles['CustomTableHeader'])])
                
                for metric, value in metrics.items():
                    # Format the metric name to be more readable
                    formatted_metric = metric.replace('_', ' ').title()
                    metric_data.append([
                        Paragraph(formatted_metric, styles['CustomTableCell']),
                        Paragraph(str(value), styles['CustomTableCell'])
                    ])
                
                # Set the table style with cell wrapping
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
                    ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ])
                
                # Create table with appropriate column widths
                col_widths = [doc.width * 0.4, doc.width * 0.6]
                table = Table(metric_data, colWidths=col_widths)
                table.setStyle(table_style)
                content.append(table)
        
        content.append(Spacer(1, 12))
        content.append(PageBreak())
    
    # Add each analysis section
    section_num = 3
    for section in ['Company Overview', 'Product Line', 'Features', 'Pricing', 
                  'Customer Reviews & Segments', 'SWOT Analysis', 
                  'Market Position', 'Technology & Innovation']:
        
        content.append(Paragraph(f"{section_num}. {section} Analysis", styles['CustomHeading1']))
        content.append(Spacer(1, 12))
        
        # Introduction to the section based on type
        section_intro = ""
        if section == "Company Overview":
            section_intro = "This section provides detailed information about each competitor's company background, history, size, and overall market presence."
        elif section == "Product Line":
            section_intro = "This section examines the product offerings of each competitor, including their main products, features, and target markets."
        elif section == "Features":
            section_intro = "This section details the specific features and capabilities offered by each competitor's products and services."
        elif section == "Pricing":
            section_intro = "This section analyzes the pricing strategies, models, and tiers used by each competitor."
        elif section == "Customer Reviews & Segments":
            section_intro = "This section examines customer feedback and the primary customer segments targeted by each competitor."
        elif section == "SWOT Analysis":
            section_intro = "This section provides a detailed analysis of each competitor's Strengths, Weaknesses, Opportunities, and Threats."
        elif section == "Market Position":
            section_intro = "This section evaluates each competitor's position in the market, including market share and competitive strategy."
        elif section == "Technology & Innovation":
            section_intro = "This section examines each competitor's technological capabilities, innovation pipeline, and R&D focus."
        
        content.append(Paragraph(section_intro, styles['CustomNormal']))
        content.append(Spacer(1, 12))
        
        # Add individual competitor analysis for this section
        for data in all_competitor_data:
            competitor = data['Competitor']
            content.append(Paragraph(f"{competitor} - {section}", styles['CustomHeading2']))
            
            # Get the analysis text
            analysis_text = ""
            structured_key = f"{section}_Structured"
            
            if structured_key in data and 'MAIN_TEXT' in data[structured_key]:
                analysis_text = data[structured_key]['MAIN_TEXT']
            else:
                analysis_text = data.get(section, f"No detailed information available for {competitor}'s {section.lower()}.")
            
            # Clean the text
            analysis_text = clean_text_for_display(analysis_text)
            
            # Split into paragraphs and add to content
            paragraphs = analysis_text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    content.append(Paragraph(paragraph.strip(), styles['CustomNormal']))
            
            # Add section-specific metrics and tables if available
            if structured_key in data:
                # Add key metrics if available
                if 'KEY_METRICS' in data[structured_key] and data[structured_key]['KEY_METRICS']:
                    metrics = data[structured_key]['KEY_METRICS']
                    content.append(Spacer(1, 6))
                    content.append(Paragraph(f"Key Metrics for {competitor} - {section}:", styles['CustomHeading2']))
                    
                    # Create a table for the metrics with appropriate styling for this section
                    metric_data = []
                    metric_data.append([
                        Paragraph("Metric", styles['CustomTableHeader']), 
                        Paragraph("Value", styles['CustomTableHeader'])
                    ])
                    
                    for metric, value in metrics.items():
                        # Format the metric name to be more readable
                        formatted_metric = metric.replace('_', ' ').title()
                        metric_data.append([
                            Paragraph(formatted_metric, styles['CustomTableCell']),
                            Paragraph(str(value), styles['CustomTableCell'])
                        ])
                    
                    # Set the table style
                    metrics_style = TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), colors.darkblue),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 6),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ])
                    
                    # Create table with appropriate column widths
                    col_widths = [doc.width * 0.4, doc.width * 0.6]
                    metrics_table = Table(metric_data, colWidths=col_widths)
                    metrics_table.setStyle(metrics_style)
                    content.append(metrics_table)
                
                # Add comparison table if available
                if 'COMPARISON_TABLE' in data[structured_key] and data[structured_key]['COMPARISON_TABLE']:
                    table_text = data[structured_key]['COMPARISON_TABLE']
                    # Check if it's a proper markdown table
                    if '|' in table_text and '-' in table_text:
                        content.append(Spacer(1, 12))
                        content.append(Paragraph(f"Comparison Table for {competitor} - {section}:", styles['CustomHeading2']))
                        
                        try:
                            # Process markdown table
                            lines = table_text.strip().split('\n')
                            headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
                            
                            # Skip the separator line
                            table_data = []
                            table_data.append([Paragraph(header, styles['CustomTableHeader']) for header in headers])
                            
                            # Add data rows
                            for line in lines[2:]:  # Skip header and separator
                                if '|' in line:
                                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                                    table_data.append([Paragraph(clean_text_for_display(cell), styles['CustomTableCell']) for cell in cells])
                            
                            # Define table style
                            comparison_style = TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                                ('TOPPADDING', (0, 0), (-1, -1), 4),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                            ])
                            
                            # Calculate column widths based on available space and number of columns
                            num_cols = len(headers)
                            if num_cols > 0:
                                col_width = doc.width / num_cols
                                col_widths = [col_width] * num_cols
                                
                                # Create the table
                                comparison_table = Table(table_data, colWidths=col_widths)
                                comparison_table.setStyle(comparison_style)
                                content.append(comparison_table)
                            else:
                                # Fallback: Add as regular text
                                content.append(Paragraph(clean_text_for_display(table_text), styles['CustomNormal']))
                        except Exception as e:
                            # If there's an error formatting the table, add as text
                            logger.warning(f"Error formatting comparison table: {str(e)}")
                            content.append(Paragraph(clean_text_for_display(table_text), styles['CustomNormal']))
                    else:
                        # Not a valid table, add as regular text
                        content.append(Paragraph(clean_text_for_display(table_text), styles['CustomNormal']))
            
            # Special handling for SWOT Analysis - use a custom table format
            if section == "SWOT Analysis" and structured_key in data:
                # Create a SWOT table
                try:
                    # Extract SWOT components
                    swot_text = analysis_text.lower()
                    
                    # Lists to store SWOT items
                    strengths = []
                    weaknesses = []
                    opportunities = []
                    threats = []
                    
                    # Extract from text using common patterns
                    lines = analysis_text.split('\n')
                    current_section = None
                    
                    # Process each line to find and categorize SWOT items
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        lower_line = line.lower()
                        
                        # Check for section headers
                        if 'strength' in lower_line and len(line) < 30:
                            current_section = 'strengths'
                            continue
                        elif 'weakness' in lower_line and len(line) < 30:
                            current_section = 'weaknesses'
                            continue
                        elif 'opportunit' in lower_line and len(line) < 30:
                            current_section = 'opportunities'
                            continue
                        elif 'threat' in lower_line and len(line) < 30:
                            current_section = 'threats'
                            continue
                            
                        # Process bullet points and numbered items
                        if current_section:
                            # Check for bullet points
                            is_bullet = False
                            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                                is_bullet = True
                            
                            # Check for numbered items
                            is_numbered = False
                            if len(line) > 2 and line[0].isdigit():
                                if line[1:3] in ['. ', ') ']:
                                    is_numbered = True
                            
                            # If we found a bullet or numbered item
                            if is_bullet or is_numbered:
                                # Clean the item text
                                item = line.lstrip('-•* 0123456789.) ')
                                
                                # Add to appropriate list
                                if current_section == 'strengths':
                                    strengths.append(item)
                                elif current_section == 'weaknesses':
                                    weaknesses.append(item)
                                elif current_section == 'opportunities':
                                    opportunities.append(item)
                                elif current_section == 'threats':
                                    threats.append(item)
                    
                    # If we have SWOT items, create a nice table
                    if strengths or weaknesses or opportunities or threats:
                        content.append(Spacer(1, 12))
                        content.append(Paragraph(f"SWOT Analysis for {competitor}:", styles['CustomHeading2']))
                        
                        # SWOT Table data
                        swot_data = [
                            [Paragraph("Strengths", styles['CustomTableHeader']), 
                             Paragraph("Weaknesses", styles['CustomTableHeader'])],
                        ]
                        
                        # Determine how many rows we need for S-W
                        max_sw_rows = max(len(strengths) or 1, len(weaknesses) or 1)
                        for i in range(max_sw_rows):
                            row = []
                            # Add strength
                            if i < len(strengths):
                                row.append(Paragraph(strengths[i], styles['CustomTableCell']))
                            else:
                                row.append(Paragraph("", styles['CustomTableCell']))
                            
                            # Add weakness
                            if i < len(weaknesses):
                                row.append(Paragraph(weaknesses[i], styles['CustomTableCell']))
                            else:
                                row.append(Paragraph("", styles['CustomTableCell']))
                                
                            swot_data.append(row)
                        
                        # Add O-T header
                        swot_data.append([
                            Paragraph("Opportunities", styles['CustomTableHeader']), 
                            Paragraph("Threats", styles['CustomTableHeader'])
                        ])
                        
                        # Add O-T content
                        max_ot_rows = max(len(opportunities) or 1, len(threats) or 1)
                        for i in range(max_ot_rows):
                            row = []
                            # Add opportunity
                            if i < len(opportunities):
                                row.append(Paragraph(opportunities[i], styles['CustomTableCell']))
                            else:
                                row.append(Paragraph("", styles['CustomTableCell']))
                            
                            # Add threat
                            if i < len(threats):
                                row.append(Paragraph(threats[i], styles['CustomTableCell']))
                            else:
                                row.append(Paragraph("", styles['CustomTableCell']))
                                
                            swot_data.append(row)
                        
                        # SWOT table style
                        swot_style = TableStyle([
                            # S-W header
                            ('BACKGROUND', (0, 0), (0, 0), colors.lightgreen),
                            ('BACKGROUND', (1, 0), (1, 0), colors.salmon),
                            # O-T header 
                            ('BACKGROUND', (0, max_sw_rows+1), (0, max_sw_rows+1), colors.lightyellow),
                            ('BACKGROUND', (1, max_sw_rows+1), (1, max_sw_rows+1), colors.lightgrey),
                            # Global formatting
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('TEXTCOLOR', (0, max_sw_rows+1), (-1, max_sw_rows+1), colors.black),
                            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                            ('ALIGN', (0, max_sw_rows+1), (-1, max_sw_rows+1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTNAME', (0, max_sw_rows+1), (-1, max_sw_rows+1), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                            ('BOTTOMPADDING', (0, max_sw_rows+1), (-1, max_sw_rows+1), 6),
                            ('BACKGROUND', (0, 1), (-1, max_sw_rows), colors.white),
                            ('BACKGROUND', (0, max_sw_rows+2), (-1, -1), colors.white),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('LEFTPADDING', (0, 0), (-1, -1), 6),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                            ('TOPPADDING', (0, 0), (-1, -1), 4),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                        ])
                        
                        # Create SWOT table with equal column widths
                        col_widths = [doc.width * 0.5, doc.width * 0.5]
                        swot_table = Table(swot_data, colWidths=col_widths)
                        swot_table.setStyle(swot_style)
                        content.append(swot_table)
                except Exception as e:
                    logger.warning(f"Error creating SWOT table: {str(e)}")
                    # Fallback: no special handling if there's an error
                    pass
            
            content.append(Spacer(1, 12))
            content.append(PageBreak())
            
        section_num += 1
    
    # Conclusion
    content.append(Paragraph(f"{section_num}. Conclusion and Recommendations", styles['CustomHeading1']))
    content.append(Spacer(1, 12))
    
    conclusion_text = f"""
    Based on the comprehensive analysis of the competitors in this report, several key strategic 
    implications emerge for {company_name}. The competitive landscape shows varying strengths 
    and weaknesses among the analyzed companies, which presents both challenges and opportunities.
    
    The Excel file accompanying this report ({excel_filename}) contains detailed data and metrics 
    that can be further analyzed to inform strategic decision-making.
    
    For detailed recommendations based on this analysis, please refer to the Executive Summary section
    and the individual SWOT analyses for each competitor.
    """
    
    conclusion_text = clean_text_for_display(conclusion_text)
    content.append(Paragraph(conclusion_text, styles['CustomNormal']))
    
    # Build the PDF
    doc.build(content)
    
    return pdf_filename

# -----------------------------------------------------------------------------
# SWOT Table Creation Function for Excel
# -----------------------------------------------------------------------------
def create_swot_table(worksheet, data, row, competitor_name, format_settings):
    """
    Create a properly formatted SWOT analysis table in Excel.
    
    Args:
        worksheet: Excel worksheet object
        data: Data dictionary containing SWOT analysis
        row: Starting row for the table
        competitor_name: Name of the competitor
        format_settings: Dictionary with formatting objects
        
    Returns:
        The next row after the SWOT table
    """
    # Unpack formats
    title_format = format_settings['title_format']
    text_format = format_settings['text_format']
    
    # Create custom formats for each SWOT quadrant
    workbook = format_settings['workbook']
    strengths_format = workbook.add_format({
        'bg_color': '#C6EFCE', 'border': 1, 'text_wrap': True, 'valign': 'top'
    })
    weaknesses_format = workbook.add_format({
        'bg_color': '#FFCCCC', 'border': 1, 'text_wrap': True, 'valign': 'top'
    })
    opportunities_format = workbook.add_format({
        'bg_color': '#FFEB9C', 'border': 1, 'text_wrap': True, 'valign': 'top'
    })
    threats_format = workbook.add_format({
        'bg_color': '#D9D9D9', 'border': 1, 'text_wrap': True, 'valign': 'top'
    })
    swot_header_format = workbook.add_format({
        'bold': True, 'bg_color': '#B7DEE8', 'border': 1, 
        'align': 'center', 'valign': 'vcenter'
    })
    
    # Add the SWOT title
    worksheet.merge_range(row, 0, row, 5, f"SWOT Analysis: {competitor_name}", title_format)
    row += 1
    
    # Extract SWOT content from the data
    structured_key = "SWOT Analysis_Structured"
    main_text = ""
    if structured_key in data and 'MAIN_TEXT' in data[structured_key]:
        main_text = data[structured_key]['MAIN_TEXT']
    else:
        main_text = data.get("SWOT Analysis", "")
    
    # Clean the text
    main_text = clean_text_for_display(main_text)
    
    # Try to extract strengths, weaknesses, opportunities, and threats
    strengths = []
    weaknesses = []
    opportunities = []
    threats = []
    
    # Look for section headers and extract content
    sections = {}
    current_section = None
    
    # First try to find explicitly labeled sections
    lines = main_text.split('\n')
    for line in lines:
        # Check for section headers
        lower_line = line.lower().strip()
        if "strength" in lower_line and len(lower_line) < 50:
            current_section = "strengths"
            sections[current_section] = []
        elif "weakness" in lower_line and len(lower_line) < 50:
            current_section = "weaknesses"
            sections[current_section] = []
        elif "opportunit" in lower_line and len(lower_line) < 50:
            current_section = "opportunities"
            sections[current_section] = []
        elif "threat" in lower_line and len(lower_line) < 50:
            current_section = "threats"
            sections[current_section] = []
        elif current_section and line.strip():
            # If we're in a section and line isn't a section header, add it
            if line.strip().startswith('-') or line.strip().startswith('•') or line.strip().startswith('*') or (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                # This is a bullet point - extract just the content
                content = line.strip()
                content = re.sub(r'^[•\-*]\s*', '', content)
                sections[current_section].append(content)
            elif sections[current_section] and not line.strip().endswith(':'):
                # Append to the last item if it's a continuation
                sections[current_section][-1] += " " + line.strip()
    
    # Extract the points
    strengths = sections.get("strengths", [])
    weaknesses = sections.get("weaknesses", [])
    opportunities = sections.get("opportunities", [])
    threats = sections.get("threats", [])
    
    # If we couldn't find explicit sections, look for bullet points and categorize
    if not (strengths or weaknesses or opportunities or threats):
        # Extract all bullet points
        bullet_points = re.findall(r'(?:^|\n)[*\-•]+(.*?)(?=(?:\n[*\-•]+|\n\n|$))', main_text, re.DOTALL)
        
        for point in bullet_points:
            point_text = point.strip()
            if not point_text:
                continue
                
            # Try to categorize based on context
            lower_point = point_text.lower()
            
            if "strength" in lower_point and ":" in point_text:
                # This is likely a strength header with content
                strengths.append(point_text.split(":", 1)[1].strip())
            elif "weakness" in lower_point and ":" in point_text:
                weaknesses.append(point_text.split(":", 1)[1].strip())
            elif "opportunit" in lower_point and ":" in point_text:
                opportunities.append(point_text.split(":", 1)[1].strip())
            elif "threat" in lower_point and ":" in point_text:
                threats.append(point_text.split(":", 1)[1].strip())
            elif any(keyword in lower_point for keyword in ["strong", "advantage", "excel", "best"]):
                strengths.append(point_text)
            elif any(keyword in lower_point for keyword in ["weak", "disadvantage", "poor", "lack"]):
                weaknesses.append(point_text)
            elif any(keyword in lower_point for keyword in ["opportunit", "potential", "future", "growth"]):
                opportunities.append(point_text)
            elif any(keyword in lower_point for keyword in ["threat", "risk", "challenge", "competition"]):
                threats.append(point_text)
    
    # Create the SWOT template header
    worksheet.write(row, 0, "Strengths", swot_header_format)
    worksheet.write(row, 3, "Weaknesses", swot_header_format)
    row += 1
    
    # Get max number of items in strengths or weaknesses for row sizing
    max_sw_rows = max(len(strengths) or 1, len(weaknesses) or 1)
    
    # Fill in strengths and weaknesses
    for i in range(max_sw_rows):
        strength_text = strengths[i] if i < len(strengths) else ""
        weakness_text = weaknesses[i] if i < len(weaknesses) else ""
        
        if strength_text:
            worksheet.merge_range(row, 0, row, 2, f"• {strength_text}", strengths_format)
        else:
            worksheet.merge_range(row, 0, row, 2, "", strengths_format)
            
        if weakness_text:
            worksheet.merge_range(row, 3, row, 5, f"• {weakness_text}", weaknesses_format)
        else:
            worksheet.merge_range(row, 3, row, 5, "", weaknesses_format)
        
        row += 1
    
    # Add a small spacer
    row += 1
    
    # Add opportunities and threats header
    worksheet.write(row, 0, "Opportunities", swot_header_format)
    worksheet.write(row, 3, "Threats", swot_header_format)
    row += 1
    
    # Get max number of items in opportunities or threats
    max_ot_rows = max(len(opportunities) or 1, len(threats) or 1)
    
    # Fill in opportunities and threats
    for i in range(max_ot_rows):
        opportunity_text = opportunities[i] if i < len(opportunities) else ""
        threat_text = threats[i] if i < len(threats) else ""
        
        if opportunity_text:
            worksheet.merge_range(row, 0, row, 2, f"• {opportunity_text}", opportunities_format)
        else:
            worksheet.merge_range(row, 0, row, 2, "", opportunities_format)
            
        if threat_text:
            worksheet.merge_range(row, 3, row, 5, f"• {threat_text}", threats_format)
        else:
            worksheet.merge_range(row, 3, row, 5, "", threats_format)
        
        row += 1
    
    # Add a spacer after the SWOT analysis
    row += 2
    
    return row

# -----------------------------------------------------------------------------
# Main Runner Function
# -----------------------------------------------------------------------------
async def main():
    """
    Main asynchronous function that orchestrates the entire process.
    """
    try:
        # Get and validate user inputs
        inputs = gather_user_inputs()
        competitors = inputs.competitors
        company_name = inputs.company_name
        
        print(f"\nAnalyzing {len(competitors)} competitors: {', '.join(competitors)}")
        
        # Initialize progress tracking
        progress = ProgressTracker(len(competitors), competitors)
        
        # List to hold competitor data
        all_competitor_data = []
        
        # Process each competitor sequentially for reliability
        for competitor in competitors:
            progress.start_competitor(competitor)
            try:
                data = await fetch_competitor_data(competitor)
                all_competitor_data.append(data)
            except Exception as e:
                logger.error(f"Failed to analyze {competitor}: {str(e)}")
                logger.error(traceback.format_exc())
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
            finally:
                progress.complete_competitor(competitor)
        
        # Mark progress as complete
        progress.complete()
        
        if not all_competitor_data:
            logger.error("No data was collected for any competitors.")
            print("\nNo data was collected. Please check the logs and try again.")
            return
        
        # Format and save to Excel
        try:
            print("\nGenerating Excel report...")
            excel_filename = format_excel_output(all_competitor_data, company_name)
            logger.info(f"Excel file generated successfully: {excel_filename}")
            
            # Generate PDF report
            print("\nGenerating PDF report...")
            pdf_filename = generate_pdf_report(all_competitor_data, company_name, excel_filename)
            logger.info(f"PDF file generated successfully: {pdf_filename}")
            
            print(f"\nReports generated successfully:")
            print(f"1. Excel report: {excel_filename}")
            print(f"2. PDF report: {pdf_filename}")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {str(e)}")
            logger.error(traceback.format_exc())
            print("An error occurred while generating the reports. Please check the logs for details.")
    
    except Exception as e:
        logger.critical(f"Critical error in execution: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"\nCritical error: {str(e)}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        logger.critical(f"Critical error in execution: {str(e)}")
        print(f"\nAn unexpected error occurred: {str(e)}") 