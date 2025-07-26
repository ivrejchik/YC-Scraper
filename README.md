# YC Company Parser

A comprehensive tool to fetch and analyze Y Combinator companies across different batches with advanced filtering, sorting, and LinkedIn integration.

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

## Command Line Interface

The parser includes a powerful CLI for batch operations and automation:

### Basic Usage

```bash
# Process companies for default batch (S25)
python -m yc_parser process

# Process companies for specific batch
python -m yc_parser process --batch W25
python -m yc_parser process --batch S24

# Show statistics for a batch
python -m yc_parser stats --batch W25

# Validate data integrity
python -m yc_parser validate --batch S25 --repair

# Create backup
python -m yc_parser backup --batch W25 --name "pre_update_backup"

# Discover LinkedIn profiles
python -m yc_parser discover --batch S25 --dry-run
```

### Available Commands

| Command | Description | Batch Support |
|---------|-------------|---------------|
| `process` | Process new companies and update dataset | ✅ `--batch` |
| `stats` | Show dataset statistics | ✅ `--batch` |
| `validate` | Validate data integrity with optional repair | ✅ `--batch` |
| `backup` | Create data backup | ✅ `--batch` |
| `discover` | Discover LinkedIn profiles for YC companies | ✅ `--batch` |
| `clear-resume` | Clear resume data from interrupted sessions | ✅ `--batch` |
| `health` | Run system health checks | ❌ |

### Batch Code Format

- **S25** = Summer 2025
- **W25** = Winter 2025  
- **S24** = Summer 2024
- **W24** = Winter 2024
- And so on...

### Examples

```bash
# Process Winter 2024 companies
python -m yc_parser process --batch W24 --verbose

# Get stats for Summer 2023
python -m yc_parser stats --batch S23

# Backup current S25 data before major update
python -m yc_parser backup --batch S25 --name "before_linkedin_update"

# Validate and repair W25 data
python -m yc_parser validate --batch W25 --repair

# Test LinkedIn discovery for S24 without saving
python -m yc_parser discover --batch S24 --dry-run --verbose
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

