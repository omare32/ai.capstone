#!/usr/bin/env python
"""
Exploratory Data Analysis Module for AAVAIL Revenue Prediction
Handles EDA, hypothesis testing, and data investigation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """
    Comprehensive EDA analysis for AAVAIL revenue prediction project
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA analyzer
        
        Args:
            df: Processed transaction DataFrame
        """
        self.df = df.copy()
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def generate_data_summary(self) -> Dict:
        """
        Generate comprehensive data summary
        
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'basic_info': {
                'total_records': len(self.df),
                'total_revenue': self.df['price'].sum(),
                'date_range': {
                    'start': self.df['date'].min(),
                    'end': self.df['date'].max(),
                    'days': (self.df['date'].max() - self.df['date'].min()).days
                },
                'unique_customers': self.df['customer_id'].nunique(),
                'unique_countries': self.df['country'].nunique(),
                'unique_invoices': self.df['invoice'].nunique()
            },
            'revenue_stats': {
                'mean_transaction': self.df['price'].mean(),
                'median_transaction': self.df['price'].median(),
                'std_transaction': self.df['price'].std(),
                'min_transaction': self.df['price'].min(),
                'max_transaction': self.df['price'].max()
            },
            'missing_data': self.df.isnull().sum().to_dict()
        }
        
        return summary
    
    def analyze_revenue_by_country(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze revenue by country and identify top performers
        
        Args:
            top_n: Number of top countries to return
            
        Returns:
            DataFrame: Country revenue analysis
        """
        country_revenue = self.df.groupby('country').agg({
            'price': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'date': ['min', 'max']
        }).round(2)
        
        country_revenue.columns = ['total_revenue', 'avg_transaction', 'total_transactions', 
                                 'unique_customers', 'first_transaction', 'last_transaction']
        
        country_revenue = country_revenue.sort_values('total_revenue', ascending=False)
        country_revenue['revenue_percentage'] = (country_revenue['total_revenue'] / 
                                               country_revenue['total_revenue'].sum() * 100).round(2)
        
        return country_revenue.head(top_n)
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in the data
        
        Returns:
            dict: Temporal analysis results
        """
        # Monthly revenue trends
        monthly_revenue = self.df.groupby('month_year')['price'].agg(['sum', 'count', 'mean']).reset_index()
        monthly_revenue.columns = ['month_year', 'total_revenue', 'transaction_count', 'avg_transaction']
        
        # Daily patterns
        daily_patterns = self.df.groupby('day_of_week')['price'].agg(['sum', 'count', 'mean'])
        daily_patterns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Quarterly patterns
        quarterly_revenue = self.df.groupby('quarter')['price'].agg(['sum', 'count', 'mean'])
        
        return {
            'monthly_trends': monthly_revenue,
            'daily_patterns': daily_patterns,
            'quarterly_patterns': quarterly_revenue
        }
    
    def analyze_customer_segments(self) -> pd.DataFrame:
        """
        Analyze customer behavior and segmentation
        
        Returns:
            DataFrame: Customer analysis
        """
        customer_analysis = self.df.groupby('customer_id').agg({
            'price': ['sum', 'mean', 'count'],
            'date': ['min', 'max'],
            'country': 'first',
            'times_viewed': 'mean'
        }).round(2)
        
        customer_analysis.columns = ['total_spent', 'avg_transaction', 'total_transactions',
                                   'first_purchase', 'last_purchase', 'country', 'avg_views']
        
        # Calculate customer lifetime
        customer_analysis['days_active'] = (customer_analysis['last_purchase'] - 
                                          customer_analysis['first_purchase']).dt.days
        
        # Customer value segmentation
        customer_analysis['customer_value'] = pd.qcut(customer_analysis['total_spent'], 
                                                     q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        return customer_analysis
    
    def identify_top_countries(self, n: int = 10) -> List[str]:
        """
        Identify top N countries by revenue for model focus
        
        Args:
            n: Number of top countries to return
            
        Returns:
            list: Top country names
        """
        country_revenue = self.df.groupby('country')['price'].sum().sort_values(ascending=False)
        return country_revenue.head(n).index.tolist()
    
    def create_visualization_suite(self, output_dir: str = "../reports/figures/"):
        """
        Create comprehensive visualization suite
        
        Args:
            output_dir: Directory to save figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Revenue by Country (Top 10)
        plt.figure(figsize=(14, 8))
        top_countries = self.analyze_revenue_by_country(10)
        bars = plt.bar(range(len(top_countries)), top_countries['total_revenue'])
        plt.title('Top 10 Countries by Total Revenue', fontsize=16, fontweight='bold')
        plt.xlabel('Country')
        plt.ylabel('Total Revenue')
        plt.xticks(range(len(top_countries)), top_countries.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}revenue_by_country.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Monthly Revenue Trends
        temporal_data = self.analyze_temporal_patterns()
        monthly_data = temporal_data['monthly_trends']
        
        plt.figure(figsize=(14, 8))
        plt.plot(monthly_data['month_year'].astype(str), monthly_data['total_revenue'], 
                marker='o', linewidth=2, markersize=6)
        plt.title('Monthly Revenue Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Month-Year')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}monthly_revenue_trends.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Transaction Amount Distribution
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['price'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.df['price'])
        plt.title('Transaction Amount Box Plot')
        plt.ylabel('Transaction Amount')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}transaction_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Daily Patterns
        daily_data = temporal_data['daily_patterns']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(daily_data.index, daily_data['sum'])
        plt.title('Revenue by Day of Week', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}daily_revenue_patterns.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Customer Segmentation
        customer_data = self.analyze_customer_segments()
        
        plt.figure(figsize=(12, 8))
        customer_value_counts = customer_data['customer_value'].value_counts()
        colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold']
        wedges, texts, autotexts = plt.pie(customer_value_counts.values, 
                                          labels=customer_value_counts.index,
                                          autopct='%1.1f%%', 
                                          colors=colors,
                                          startangle=90)
        plt.title('Customer Value Segmentation', fontsize=16, fontweight='bold')
        plt.savefig(f"{output_dir}customer_segmentation.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. Revenue Heatmap by Country and Month
        top_10_countries = self.identify_top_countries(10)
        country_month_pivot = self.df[self.df['country'].isin(top_10_countries)].pivot_table(
            values='price', index='country', columns='month_year', aggfunc='sum', fill_value=0)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(country_month_pivot, annot=False, cmap='YlOrRd', fmt='.0f')
        plt.title('Revenue Heatmap: Top 10 Countries by Month', fontsize=16, fontweight='bold')
        plt.xlabel('Month-Year')
        plt.ylabel('Country')
        plt.tight_layout()
        plt.savefig(f"{output_dir}revenue_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_hypotheses(self) -> List[str]:
        """
        Generate testable hypotheses based on business scenario
        
        Returns:
            list: Testable hypotheses
        """
        hypotheses = [
            "H1: Revenue shows seasonal patterns that can be leveraged for prediction",
            "H2: The top 10 countries by revenue contribute to 80% of total revenue (Pareto principle)",
            "H3: Customer transaction frequency correlates with total customer value",
            "H4: Monthly revenue trends show growth patterns that can be extrapolated",
            "H5: Weekend vs weekday transaction patterns differ significantly",
            "H6: Country-specific revenue patterns are stable over time",
            "H7: Customer retention (repeat purchases) impacts monthly revenue predictability",
            "H8: Transaction amount distributions vary significantly by country"
        ]
        
        return hypotheses
    
    def test_hypotheses(self) -> Dict:
        """
        Test the generated hypotheses with statistical analysis
        
        Returns:
            dict: Hypothesis testing results
        """
        results = {}
        
        # H2: Top 10 countries contribute to 80% of revenue
        country_revenue = self.df.groupby('country')['price'].sum().sort_values(ascending=False)
        total_revenue = country_revenue.sum()
        top_10_revenue = country_revenue.head(10).sum()
        top_10_percentage = (top_10_revenue / total_revenue) * 100
        
        results['pareto_principle'] = {
            'top_10_percentage': top_10_percentage,
            'passes_80_20_rule': top_10_percentage >= 80
        }
        
        # H5: Weekend vs weekday patterns
        weekend_revenue = self.df[self.df['is_weekend']]['price'].sum()
        weekday_revenue = self.df[~self.df['is_weekend']]['price'].sum()
        
        results['weekend_vs_weekday'] = {
            'weekend_revenue': weekend_revenue,
            'weekday_revenue': weekday_revenue,
            'weekend_percentage': (weekend_revenue / (weekend_revenue + weekday_revenue)) * 100
        }
        
        return results

def perform_eda(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Main function to perform comprehensive EDA
    
    Args:
        df: Processed transaction DataFrame
        
    Returns:
        tuple: (eda_results, hypothesis_tests)
    """
    analyzer = EDAAnalyzer(df)
    
    # Generate comprehensive analysis
    data_summary = analyzer.generate_data_summary()
    country_analysis = analyzer.analyze_revenue_by_country()
    temporal_analysis = analyzer.analyze_temporal_patterns()
    customer_analysis = analyzer.analyze_customer_segments()
    hypotheses = analyzer.generate_hypotheses()
    hypothesis_tests = analyzer.test_hypotheses()
    
    # Create visualizations
    analyzer.create_visualization_suite()
    
    eda_results = {
        'data_summary': data_summary,
        'country_analysis': country_analysis,
        'temporal_analysis': temporal_analysis,
        'customer_analysis': customer_analysis,
        'hypotheses': hypotheses,
        'top_countries': analyzer.identify_top_countries()
    }
    
    return eda_results, hypothesis_tests
