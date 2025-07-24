# YC Company Parser

A comprehensive tool to fetch and analyze Y Combinator companies with advanced filtering, sorting, and LinkedIn integration.

## Quick Start

### **Live Demo**
**Try it now**: [YC Company Parser on Streamlit Cloud](https://yc-scraper-test.streamlit.app)

### **Local Development**

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the App
```bash
streamlit run app.py
```

#### 3. Use the Interface
- Open your browser to `http://localhost:8501`
- Select **Season** (Summer/Winter) and **Year** (2025-2020)
- Click **"Refresh Data"** to fetch YC companies
- Use filters and sorting to explore the data
- Download filtered results as CSV



## What It Does

- **Fetches YC companies** from official YC API for any batch
- **Extracts LinkedIn URLs** from YC profile pages  
- **Checks LinkedIn profiles** for YC mentions
- **Advanced filtering & sorting** with real-time search
- **Export functionality** with custom filenames

## Features

### **Batch Selection**
- **Season Dropdown**: Summer or Winter batches
- **Year Selection**: 2025 down to 2020
- **Batch Display**: Shows selected batch code (e.g., S25, W24)
- **API Compatibility**: Warns when selected batch may not be available

### **Advanced Filtering**
- **🔍 Text Search**: Search across company names, descriptions, and websites
- **🔗 LinkedIn Filter**: All, Has LinkedIn, No LinkedIn, YC Mention, No YC Mention
- **📊 Smart Sorting**: Name A-Z/Z-A, YC Mentions First, LinkedIn First, Recently Updated

### **Data Export**
- **CSV Download**: Export filtered data with dynamic filenames
- **Custom Naming**: Files named by batch and date (e.g., `yc_s25_companies_20250724.csv`)
- **Filtered Results**: Download only the data you're viewing

### **LinkedIn Integration**
- **URL Extraction**: Scrapes LinkedIn URLs from YC profile pages
- **Mention Detection**: Checks if companies mention their YC batch on LinkedIn
- **Flexible Validation**: Accepts various LinkedIn URL formats
- **Rate Limiting**: Handles LinkedIn's anti-bot protection gracefully

## 📁 Project Structure

```
├── app.py                    # Streamlit web interface with all features
├── requirements.txt          # Python dependencies
├── yc_parser/               # Core parsing logic
│   ├── yc_client.py         # YC API client
│   ├── linkedin_scraper.py  # LinkedIn integration
│   ├── company_processor.py # Main processing pipeline
│   ├── data_manager.py      # Data persistence
│   └── models.py            # Data models and validation
├── test_*.py                # Unit tests
└── yc_s25_companies.csv     # Generated data file
```

## Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_yc_client.py -v
```

## Data Output

The tool generates CSV files with:
- Company name and website
- YC profile URL and description
- LinkedIn URL (if found)
- YC mention flag (if company promotes YC status)
- Last updated timestamp

