# AAVAIL Revenue Prediction - AI Workflow Capstone

## Project Overview

This project implements a machine learning solution for AAVAIL to predict monthly revenue using transaction-level data from their à la carte billing model across 38 countries.

## Business Context

AAVAIL is transitioning from a tiered subscription model to an à la carte approach. Management needs a service that can predict monthly revenue at any point in time, with the ability to project revenue for specific countries. The focus is on the top 10 countries by revenue.

## Project Structure

```
├── data/                    # Data files (JSON format by month)
├── src/                     # Source code
│   ├── data_ingestion.py   # Data loading and processing functions
│   ├── eda.py              # Exploratory Data Analysis
│   └── visualization.py    # Visualization utilities
├── notebooks/              # Jupyter notebooks for analysis
├── reports/                # Deliverable reports and presentations
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Part 1: Data Investigation (Assignment 01)

### Objectives
1. Assimilate business scenario and articulate testable hypotheses
2. State ideal data requirements for addressing the business opportunity
3. Create automated data ingestion pipeline
4. Investigate relationships between data, target, and business metrics
5. Create deliverable with visualizations

### Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the data investigation notebook:
   ```bash
   jupyter notebook notebooks/part1_data_investigation.ipynb
   ```

## Data Description

- **Source**: Online retail transaction data (JSON format)
- **Time Range**: Multiple months of historical data
- **Features**: country, customer_id, day, invoice, month, price, stream_id, times_viewed, year
- **Target**: Monthly revenue prediction
- **Scope**: Focus on top 10 countries by revenue

## Technology Stack

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Machine learning
- **Jupyter** - Interactive analysis
- **Docker** - Containerization (Part 3)

## Deliverables

### Part 1: Data Investigation Report
- Business scenario analysis
- Testable hypotheses
- Data ingestion automation
- EDA findings with visualizations
- Recommendations for model development

## Author

AI Developer - AAVAIL Revenue Prediction Team
