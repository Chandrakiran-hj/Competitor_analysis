# Competitive Analysis Report Generator

An AI-powered tool that generates comprehensive competitor analysis reports using OpenAI's Agents SDK. The tool provides both Excel and PDF reports with detailed insights about your competitors.

## Features

- Real-time competitor analysis using AI
- Beautiful and intuitive web interface
- Progress tracking and live updates
- Preview of generated reports
- Downloadable Excel and PDF reports
- Comprehensive analysis including:
  - Company Overview
  - Product Lines
  - Features
  - Pricing
  - Customer Reviews
  - SWOT Analysis
  - Market Position

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Enter your company name and competitor names
4. Click "Generate Analysis" and wait for the results
5. Preview and download the generated reports

## Notes

- Each competitor analysis takes approximately 2-4 minutes
- The tool is limited to analyzing up to 10 competitors at a time
- Make sure you have a stable internet connection for the analysis
- The OpenAI API key must have sufficient credits for the analysis

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Internet connection
- Modern web browser

## License

MIT License 