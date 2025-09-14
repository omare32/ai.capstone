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
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def generate_data_summary(self) -> Dict:
        """
        Generate comprehensive data summary
        
        Returns:
            dict: Data summary statistics
        """
        self.df['year'] = self.df['InvoiceDate'].dt.year
        self.df['month'] = self.df['InvoiceDate'].dt.month
        self.df['quarter'] = self.df['InvoiceDate'].dt.quarter
        self.df['day_of_week'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'] >= 5
        
        summary = {
            'basic_info': {
                'total_records': len(self.df),
                'total_revenue': (self.df['Quantity'] * self.df['UnitPrice']).sum(),
                'date_range': {
                    'start': self.df['InvoiceDate'].min(),
                    'end': self.df['InvoiceDate'].max(),
                    'days': (self.df['InvoiceDate'].max() - self.df['InvoiceDate'].min()).days
                },
                'unique_customers': self.df['CustomerID'].nunique(),
                'unique_countries': self.df['Country'].nunique(),
                'unique_invoices': self.df['InvoiceNo'].nunique()
            },
            'revenue_stats': {
                'mean_transaction': (self.df['Quantity'] * self.df['UnitPrice']).mean(),
                'median_transaction': (self.df['Quantity'] * self.df['UnitPrice']).median(),
                'std_transaction': (self.df['Quantity'] * self.df['UnitPrice']).std(),
                'min_transaction': (self.df['Quantity'] * self.df['UnitPrice']).min(),
                'max_transaction': (self.df['Quantity'] * self.df['UnitPrice']).max()
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
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        country_revenue = self.df.groupby('Country').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).sort_values(ascending=False)
        country_revenue = country_revenue.to_frame('total_revenue')
        country_revenue['avg_transaction'] = self.df.groupby('Country').apply(lambda x: (x['Quantity'] * x['UnitPrice']).mean())
        country_revenue['total_transactions'] = self.df.groupby('Country').size()
        country_revenue['unique_customers'] = self.df.groupby('Country')['CustomerID'].nunique()
        country_revenue['first_transaction'] = self.df.groupby('Country')['InvoiceDate'].min()
        country_revenue['last_transaction'] = self.df.groupby('Country')['InvoiceDate'].max()
        country_revenue['revenue_percentage'] = (country_revenue['total_revenue'] / country_revenue['total_revenue'].sum() * 100).round(2)
        
        return country_revenue.head(top_n)
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in the data
        
        Returns:
            dict: Temporal analysis results
        """
        # Monthly trends
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        monthly_revenue = revenue.groupby([self.df['year'], self.df['month']]).sum().reset_index()
        monthly_revenue.columns = ['year', 'month', 'total_revenue']
        monthly_revenue['transaction_count'] = self.df.groupby([self.df['year'], self.df['month']]).size()
        monthly_revenue['avg_transaction'] = self.df.groupby([self.df['year'], self.df['month']]).apply(lambda x: (x['Quantity'] * x['UnitPrice']).mean())
        
        # Daily patterns
        daily_patterns = revenue.groupby(self.df['day_of_week']).agg(['sum', 'count', 'mean'])
        daily_patterns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Quarterly patterns
        quarterly_revenue = revenue.groupby(self.df['quarter']).agg(['sum', 'count', 'mean'])
        
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
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        customer_analysis = self.df.groupby('CustomerID').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).to_frame('total_spent')
        customer_analysis['avg_transaction'] = self.df.groupby('CustomerID').apply(lambda x: (x['Quantity'] * x['UnitPrice']).mean())
        customer_analysis['total_transactions'] = self.df.groupby('CustomerID').size()
        customer_analysis['first_purchase'] = self.df.groupby('CustomerID')['InvoiceDate'].min()
        customer_analysis['last_purchase'] = self.df.groupby('CustomerID')['InvoiceDate'].max()
        customer_analysis['country'] = self.df.groupby('CustomerID')['Country'].first()
        customer_analysis['avg_views'] = self.df.groupby('CustomerID')['times_viewed'].mean()
        
        # Calculate customer lifetime
        customer_analysis['days_active'] = (customer_analysis['last_purchase'] - customer_analysis['first_purchase']).dt.days
        
        # Customer value segmentation
        customer_analysis['customer_value'] = pd.qcut(customer_analysis['total_spent'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        return customer_analysis
    
    def identify_top_countries(self, n: int = 10) -> List[str]:
        """
        Identify top N countries by revenue for model focus
        
        Args:
            n: Number of top countries to return
            
        Returns:
            list: Top country names
        """
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        country_revenue = self.df.groupby('Country').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).sort_values(ascending=False)
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
        plt.plot(monthly_data['month'], monthly_data['total_revenue'], 
                marker='o', linewidth=2, markersize=6)
        plt.title('Monthly Revenue Trends', fontsize=16, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Total Revenue')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}monthly_revenue_trends.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Transaction Amount Distribution
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist((self.df['Quantity'] * self.df['UnitPrice']), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.boxplot((self.df['Quantity'] * self.df['UnitPrice']))
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
        country_month_pivot = self.df[self.df['Country'].isin(top_10_countries)].pivot_table(
            values='UnitPrice', index='Country', columns='month', aggfunc='sum', fill_value=0)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(country_month_pivot, annot=False, cmap='YlOrRd', fmt='.0f')
        plt.title('Revenue Heatmap: Top 10 Countries by Month', fontsize=16, fontweight='bold')
        plt.xlabel('Month')
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
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        country_revenue = self.df.groupby('Country').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).sort_values(ascending=False)
        total_revenue = country_revenue.sum()
        top_10_revenue = country_revenue.head(10).sum()
        top_10_percentage = (top_10_revenue / total_revenue) * 100
        
        results['pareto_principle'] = {
            'top_10_percentage': top_10_percentage,
            'passes_80_20_rule': top_10_percentage >= 80
        }
        
        # H5: Weekend vs weekday analysis
        revenue = self.df['Quantity'] * self.df['UnitPrice']
        weekend_revenue = revenue[self.df['is_weekend']].sum()
        weekday_revenue = revenue[~self.df['is_weekend']].sum()
        
        results['weekend_vs_weekday'] = {
            'weekend_revenue': weekend_revenue,
            'weekday_revenue': weekday_revenue,
            'weekend_percentage': (weekend_revenue / (weekend_revenue + weekday_revenue)) * 100
        }
        
        return results
    
    def create_enhanced_visualizations(self, save_plots: bool = True):
        """Create comprehensive enhanced visualizations"""
        import os
        os.makedirs('../reports/figures', exist_ok=True)
        
        # Calculate revenue for plotting
        self.df['Revenue'] = self.df['Quantity'] * self.df['UnitPrice']
        
        # 1. Enhanced Revenue by Country with gradient colors
        plt.figure(figsize=(16, 8))
        country_revenue = self.df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(15)
        colors = plt.cm.viridis(np.linspace(0, 1, len(country_revenue)))
        
        bars = plt.bar(range(len(country_revenue)), country_revenue.values, color=colors)
        plt.title('📊 Top 15 Countries by Total Revenue\n(Enhanced Visualization)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Countries', fontsize=12)
        plt.ylabel('Revenue (€)', fontsize=12)
        plt.xticks(range(len(country_revenue)), country_revenue.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, country_revenue.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'€{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig('../reports/figures/enhanced_country_revenue.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Monthly Revenue Heatmap
        plt.figure(figsize=(16, 10))
        monthly_data = self.df.copy()
        monthly_data['YearMonth'] = monthly_data['InvoiceDate'].dt.to_period('M')
        monthly_data['Day'] = monthly_data['InvoiceDate'].dt.day
        monthly_data['Month'] = monthly_data['InvoiceDate'].dt.month
        
        # Create pivot table for heatmap
        heatmap_data = monthly_data.groupby(['Month', 'Day'])['Revenue'].sum().unstack(fill_value=0)
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.0f', 
                   cbar_kws={'label': 'Revenue (€)'}, linewidths=0.1)
        plt.title('🔥 Daily Revenue Heatmap by Month\n(Seasonal Patterns Visualization)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Day of Month', fontsize=12)
        plt.ylabel('Month', fontsize=12)
        plt.tight_layout()
        if save_plots:
            plt.savefig('../reports/figures/revenue_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Customer Segmentation Scatter Plot
        plt.figure(figsize=(14, 10))
        customer_data = self.df.dropna(subset=['CustomerID'])
        customer_metrics = customer_data.groupby('CustomerID').agg({
            'Revenue': ['sum', 'count'],
            'InvoiceDate': ['min', 'max']
        })
        customer_metrics.columns = ['TotalRevenue', 'TransactionCount', 'FirstPurchase', 'LastPurchase']
        customer_metrics['Recency'] = (customer_metrics['LastPurchase'].max() - customer_metrics['LastPurchase']).dt.days
        
        scatter = plt.scatter(customer_metrics['TransactionCount'], customer_metrics['TotalRevenue'],
                             c=customer_metrics['Recency'], s=60, alpha=0.7, cmap='RdYlBu_r')
        
        plt.xlabel('Transaction Frequency', fontsize=12)
        plt.ylabel('Customer Lifetime Value (€)', fontsize=12)
        plt.title('🎯 Customer Segmentation Analysis\n(Frequency vs Value vs Recency)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Days Since Last Purchase (Recency)', rotation=270, labelpad=20)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig('../reports/figures/customer_segmentation_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Interactive-style Time Series with Multiple Metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Revenue trends
        daily_revenue = self.df.groupby(self.df['InvoiceDate'].dt.date)['Revenue'].sum()
        ax1.plot(daily_revenue.index, daily_revenue.values, linewidth=2, color='#2E86AB')
        ax1.fill_between(daily_revenue.index, daily_revenue.values, alpha=0.3, color='#2E86AB')
        ax1.set_title('📈 Daily Revenue Trends', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Revenue (€)')
        ax1.grid(True, alpha=0.3)
        
        # Customer count trends
        daily_customers = self.df.groupby(self.df['InvoiceDate'].dt.date)['CustomerID'].nunique()
        ax2.plot(daily_customers.index, daily_customers.values, linewidth=2, color='#A23B72')
        ax2.fill_between(daily_customers.index, daily_customers.values, alpha=0.3, color='#A23B72')
        ax2.set_title('👥 Daily Unique Customers', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Unique Customers')
        ax2.grid(True, alpha=0.3)
        
        # Average order value
        daily_aov = self.df.groupby(self.df['InvoiceDate'].dt.date)['Revenue'].mean()
        ax3.plot(daily_aov.index, daily_aov.values, linewidth=2, color='#F18F01')
        ax3.fill_between(daily_aov.index, daily_aov.values, alpha=0.3, color='#F18F01')
        ax3.set_title('💰 Average Order Value', fontweight='bold', fontsize=12)
        ax3.set_ylabel('AOV (€)')
        ax3.grid(True, alpha=0.3)
        
        # Transaction volume
        daily_transactions = self.df.groupby(self.df['InvoiceDate'].dt.date).size()
        ax4.plot(daily_transactions.index, daily_transactions.values, linewidth=2, color='#C73E1D')
        ax4.fill_between(daily_transactions.index, daily_transactions.values, alpha=0.3, color='#C73E1D')
        ax4.set_title('🛒 Daily Transaction Volume', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Number of Transactions')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('🚀 AAVAIL Business Metrics Dashboard\n(Multi-dimensional Performance View)', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        if save_plots:
            plt.savefig('../reports/figures/business_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Product Analysis Visualization
        plt.figure(figsize=(16, 8))
        
        # Top products by revenue
        product_revenue = self.df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(20)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(product_revenue))
        colors = plt.cm.plasma(np.linspace(0, 1, len(product_revenue)))
        
        bars = plt.barh(y_pos, product_revenue.values, color=colors)
        plt.yticks(y_pos, [desc[:40] + '...' if len(desc) > 40 else desc for desc in product_revenue.index])
        plt.xlabel('Revenue (€)', fontsize=12)
        plt.title('🏆 Top 20 Products by Revenue\n(Best Sellers Analysis)', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels
        for bar, value in zip(bars, product_revenue.values):
            plt.text(bar.get_width() + value*0.01, bar.get_y() + bar.get_height()/2,
                    f'€{value:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig('../reports/figures/top_products_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✨ Enhanced visualizations created successfully!")
        print("📊 Saved to: ../reports/figures/")
        
        return True

def perform_eda(df: pd.DataFrame, save_plots: bool = True) -> Tuple[Dict, Dict]:
    """
    Main function to perform comprehensive EDA
    
    Args:
        df: Processed transaction DataFrame
        save_plots: Whether to save plots to files
        
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
