#!/usr/bin/env python
"""
Post-Production Analysis Script for AAVAIL Revenue Prediction
Investigates relationship between model performance and business metrics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('../src')

class PostProductionAnalyzer:
    """
    Analyzes model performance in production and its impact on business metrics
    """
    
    def __init__(self, logs_path: str = '../logs', models_path: str = '../models'):
        """
        Initialize analyzer
        
        Args:
            logs_path: Path to logs directory
            models_path: Path to models directory
        """
        self.logs_path = logs_path
        self.models_path = models_path
        self.prediction_logs = []
        self.business_metrics = {}
        
    def load_prediction_logs(self) -> pd.DataFrame:
        """
        Load prediction logs from production API
        
        Returns:
            DataFrame: Prediction logs
        """
        try:
            with open(os.path.join(self.logs_path, 'predictions.json'), 'r') as f:
                self.prediction_logs = json.load(f)
            
            if self.prediction_logs:
                df = pd.DataFrame(self.prediction_logs)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['prediction_date'] = pd.to_datetime(df['prediction_date'])
                return df
            else:
                return pd.DataFrame()
                
        except FileNotFoundError:
            print("No prediction logs found. Generating sample logs for analysis...")
            return self._generate_sample_logs()
    
    def _generate_sample_logs(self) -> pd.DataFrame:
        """
        Generate sample prediction logs for demonstration
        
        Returns:
            DataFrame: Sample prediction logs
        """
        np.random.seed(42)
        
        # Generate 100 sample predictions over 30 days
        base_date = datetime.now() - timedelta(days=30)
        sample_logs = []
        
        countries = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'Ireland']
        model_types = ['Random Forest', 'Gradient Boosting', 'ARIMA', 'LSTM']
        
        for i in range(100):
            prediction_date = base_date + timedelta(days=np.random.randint(0, 30))
            
            # Simulate realistic predictions with some variance
            base_prediction = 50000 + np.random.normal(0, 10000)
            
            # Simulate model performance metrics
            mape = np.random.uniform(0.05, 0.25)  # 5% to 25% MAPE
            mae = np.random.uniform(1000, 5000)   # MAE in euros
            
            log_entry = {
                'timestamp': (prediction_date + timedelta(hours=np.random.randint(0, 24))).isoformat(),
                'country': np.random.choice(countries),
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'prediction': base_prediction,
                'model_type': np.random.choice(model_types),
                'model_performance': {
                    'mape': mape,
                    'mae': mae
                }
            }
            
            sample_logs.append(log_entry)
        
        # Flatten model_performance for easier analysis
        flattened_logs = []
        for log in sample_logs:
            flat_log = log.copy()
            flat_log['mape'] = log['model_performance']['mape']
            flat_log['mae'] = log['model_performance']['mae']
            del flat_log['model_performance']
            flattened_logs.append(flat_log)
        
        df = pd.DataFrame(flattened_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        
        return df
    
    def simulate_actual_revenue(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate actual revenue data for comparison with predictions
        
        Args:
            prediction_df: DataFrame with predictions
            
        Returns:
            DataFrame: Predictions with simulated actual revenue
        """
        df = prediction_df.copy()
        
        # Simulate actual revenue with some noise around predictions
        # In reality, this would come from actual business data
        df['actual_revenue'] = df['prediction'] * (1 + np.random.normal(0, 0.1, len(df)))
        
        # Calculate prediction accuracy metrics
        df['absolute_error'] = np.abs(df['actual_revenue'] - df['prediction'])
        df['percentage_error'] = df['absolute_error'] / df['actual_revenue'] * 100
        
        return df
    
    def analyze_model_performance_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze model performance trends over time
        
        Args:
            df: DataFrame with predictions and actuals
            
        Returns:
            dict: Performance analysis results
        """
        # Group by model type and analyze performance
        model_performance = df.groupby('model_type').agg({
            'mape': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max'],
            'percentage_error': ['mean', 'std'],
            'prediction': 'count'
        }).round(4)
        
        # Time-based performance analysis
        df['date'] = df['timestamp'].dt.date
        daily_performance = df.groupby('date').agg({
            'percentage_error': 'mean',
            'mape': 'mean',
            'mae': 'mean',
            'prediction': 'count'
        }).round(4)
        
        # Country-specific performance
        country_performance = df.groupby('country').agg({
            'percentage_error': 'mean',
            'mape': 'mean',
            'prediction': 'sum',
            'actual_revenue': 'sum'
        }).round(2)
        
        # Performance degradation detection
        df_sorted = df.sort_values('timestamp')
        recent_performance = df_sorted.tail(20)['percentage_error'].mean()
        historical_performance = df_sorted.head(20)['percentage_error'].mean()
        
        performance_drift = {
            'recent_accuracy': recent_performance,
            'historical_accuracy': historical_performance,
            'drift_percentage': ((recent_performance - historical_performance) / historical_performance * 100) if historical_performance > 0 else 0,
            'drift_detected': abs(recent_performance - historical_performance) > historical_performance * 0.2
        }
        
        return {
            'model_performance': model_performance,
            'daily_performance': daily_performance,
            'country_performance': country_performance,
            'performance_drift': performance_drift
        }
    
    def calculate_business_impact(self, df: pd.DataFrame) -> Dict:
        """
        Calculate business impact of model predictions
        
        Args:
            df: DataFrame with predictions and actuals
            
        Returns:
            dict: Business impact metrics
        """
        # Revenue prediction accuracy impact
        total_predicted_revenue = df['prediction'].sum()
        total_actual_revenue = df['actual_revenue'].sum()
        
        revenue_accuracy = 1 - abs(total_predicted_revenue - total_actual_revenue) / total_actual_revenue
        
        # Cost of prediction errors (assuming operational costs)
        # Overestimation cost: Excess staffing/inventory
        # Underestimation cost: Missed opportunities/shortages
        
        overestimation_errors = df[df['prediction'] > df['actual_revenue']]
        underestimation_errors = df[df['prediction'] < df['actual_revenue']]
        
        # Simulate business costs (in practice, these would be real business metrics)
        overestimation_cost = overestimation_errors['absolute_error'].sum() * 0.05  # 5% of error as cost
        underestimation_cost = underestimation_errors['absolute_error'].sum() * 0.15  # 15% of error as opportunity cost
        
        total_error_cost = overestimation_cost + underestimation_cost
        
        # ROI calculation
        # Assume manual forecasting costs €1000/month per country
        countries_served = df['country'].nunique()
        manual_forecasting_cost = countries_served * 1000 * (df['timestamp'].max() - df['timestamp'].min()).days / 30
        
        cost_savings = manual_forecasting_cost - total_error_cost
        roi_percentage = (cost_savings / manual_forecasting_cost * 100) if manual_forecasting_cost > 0 else 0
        
        # Decision quality metrics
        predictions_within_10_percent = len(df[df['percentage_error'] <= 10]) / len(df) * 100
        predictions_within_20_percent = len(df[df['percentage_error'] <= 20]) / len(df) * 100
        
        return {
            'revenue_accuracy': revenue_accuracy,
            'total_predicted_revenue': total_predicted_revenue,
            'total_actual_revenue': total_actual_revenue,
            'overestimation_cost': overestimation_cost,
            'underestimation_cost': underestimation_cost,
            'total_error_cost': total_error_cost,
            'manual_forecasting_cost': manual_forecasting_cost,
            'cost_savings': cost_savings,
            'roi_percentage': roi_percentage,
            'predictions_within_10_percent': predictions_within_10_percent,
            'predictions_within_20_percent': predictions_within_20_percent,
            'average_prediction_error': df['percentage_error'].mean()
        }
    
    def generate_performance_visualizations(self, df: pd.DataFrame, analysis_results: Dict, output_dir: str = '../reports/figures'):
        """
        Generate visualizations for performance analysis
        
        Args:
            df: DataFrame with analysis data
            analysis_results: Results from performance analysis
            output_dir: Directory to save figures
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        plt.figure(figsize=(12, 8))
        model_perf = analysis_results['model_performance']['mape']['mean'].reset_index()
        bars = plt.bar(model_perf['model_type'], model_perf['mean'] * 100)
        plt.title('Model Performance Comparison (MAPE %)', fontsize=16, fontweight='bold')
        plt.xlabel('Model Type')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, model_perf['mean'] * 100):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Performance Over Time
        plt.figure(figsize=(14, 8))
        daily_perf = analysis_results['daily_performance']
        plt.plot(daily_perf.index, daily_perf['percentage_error'], marker='o', linewidth=2)
        plt.title('Model Performance Trend Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Average Percentage Error (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = np.arange(len(daily_perf))
        z = np.polyfit(x_numeric, daily_perf['percentage_error'], 1)
        p = np.poly1d(z)
        plt.plot(daily_perf.index, p(x_numeric), "--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_trend_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Prediction vs Actual Revenue
        plt.figure(figsize=(10, 8))
        plt.scatter(df['actual_revenue'], df['prediction'], alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(df['actual_revenue'].min(), df['prediction'].min())
        max_val = max(df['actual_revenue'].max(), df['prediction'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Revenue (€)')
        plt.ylabel('Predicted Revenue (€)')
        plt.title('Prediction Accuracy: Predicted vs Actual Revenue', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = df['actual_revenue'].corr(df['prediction'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Business Impact Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROI
        business_impact = analysis_results.get('business_impact', {})
        roi = business_impact.get('roi_percentage', 0)
        ax1.bar(['ROI'], [roi], color='green' if roi > 0 else 'red')
        ax1.set_title('Return on Investment (%)')
        ax1.set_ylabel('ROI (%)')
        ax1.text(0, roi/2, f'{roi:.1f}%', ha='center', va='center', fontweight='bold')
        
        # Prediction Accuracy Distribution
        ax2.hist(df['percentage_error'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(df['percentage_error'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["percentage_error"].mean():.1f}%')
        ax2.set_title('Prediction Error Distribution')
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Country Performance
        country_perf = analysis_results['country_performance']['percentage_error'].sort_values()
        ax3.barh(range(len(country_perf)), country_perf.values)
        ax3.set_yticks(range(len(country_perf)))
        ax3.set_yticklabels(country_perf.index)
        ax3.set_title('Average Error by Country (%)')
        ax3.set_xlabel('Percentage Error (%)')
        
        # Cost Analysis
        costs = [business_impact.get('overestimation_cost', 0), 
                business_impact.get('underestimation_cost', 0)]
        cost_labels = ['Overestimation\nCost', 'Underestimation\nCost']
        ax4.pie(costs, labels=cost_labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Error Cost Breakdown')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/business_impact_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, df: pd.DataFrame, analysis_results: Dict, business_impact: Dict) -> str:
        """
        Generate comprehensive post-production analysis report
        
        Args:
            df: Analysis DataFrame
            analysis_results: Performance analysis results
            business_impact: Business impact metrics
            
        Returns:
            str: Report content
        """
        report = f"""
# Post-Production Analysis Report
## AAVAIL Revenue Prediction Model Performance

**Analysis Period**: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}  
**Total Predictions**: {len(df):,}  
**Countries Served**: {df['country'].nunique()}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

The AAVAIL revenue prediction model has been in production for {(df['timestamp'].max() - df['timestamp'].min()).days} days, generating {len(df):,} predictions across {df['country'].nunique()} countries. 

**Key Performance Indicators:**
- Overall Prediction Accuracy: {100 - df['percentage_error'].mean():.1f}%
- Average Prediction Error: {df['percentage_error'].mean():.1f}%
- Revenue Prediction Accuracy: {business_impact['revenue_accuracy']*100:.1f}%
- ROI: {business_impact['roi_percentage']:.1f}%

---

## Model Performance Analysis

### Overall Performance Metrics
- **Mean Absolute Percentage Error**: {df['percentage_error'].mean():.2f}%
- **Standard Deviation of Errors**: {df['percentage_error'].std():.2f}%
- **Predictions within 10% accuracy**: {business_impact['predictions_within_10_percent']:.1f}%
- **Predictions within 20% accuracy**: {business_impact['predictions_within_20_percent']:.1f}%

### Performance by Model Type
"""
        
        # Add model performance details
        model_perf = analysis_results['model_performance']['mape']['mean']
        for model_type, mape in model_perf.items():
            report += f"- **{model_type}**: {mape*100:.2f}% MAPE\n"
        
        report += f"""
### Performance Drift Analysis
- **Recent Performance**: {analysis_results['performance_drift']['recent_accuracy']:.2f}% error
- **Historical Performance**: {analysis_results['performance_drift']['historical_accuracy']:.2f}% error
- **Drift Detected**: {'⚠️ YES' if analysis_results['performance_drift']['drift_detected'] else '✅ NO'}

---

## Business Impact Assessment

### Financial Impact
- **Total Revenue Predicted**: €{business_impact['total_predicted_revenue']:,.2f}
- **Total Actual Revenue**: €{business_impact['total_actual_revenue']:,.2f}
- **Revenue Forecast Accuracy**: {business_impact['revenue_accuracy']*100:.1f}%

### Cost Analysis
- **Manual Forecasting Cost (Baseline)**: €{business_impact['manual_forecasting_cost']:,.2f}
- **Overestimation Error Cost**: €{business_impact['overestimation_cost']:,.2f}
- **Underestimation Error Cost**: €{business_impact['underestimation_cost']:,.2f}
- **Total Error Cost**: €{business_impact['total_error_cost']:,.2f}
- **Net Cost Savings**: €{business_impact['cost_savings']:,.2f}
- **Return on Investment**: {business_impact['roi_percentage']:.1f}%

### Country-Specific Performance
"""
        
        # Add country performance
        country_perf = analysis_results['country_performance']['percentage_error'].sort_values()
        for country, error in country_perf.items():
            report += f"- **{country}**: {error:.2f}% average error\n"
        
        report += f"""
---

## Recommendations

### Model Performance
"""
        
        # Dynamic recommendations based on performance
        avg_error = df['percentage_error'].mean()
        if avg_error > 20:
            report += "- 🔴 **CRITICAL**: Average error >20%. Immediate model retraining required.\n"
        elif avg_error > 15:
            report += "- 🟡 **WARNING**: Average error >15%. Consider model retraining within 1 week.\n"
        elif avg_error > 10:
            report += "- 🟠 **ATTENTION**: Average error >10%. Monitor closely and prepare for retraining.\n"
        else:
            report += "- 🟢 **GOOD**: Average error <10%. Model performing within acceptable range.\n"
        
        if analysis_results['performance_drift']['drift_detected']:
            report += "- ⚠️ **DRIFT DETECTED**: Significant performance degradation observed. Investigate data distribution changes.\n"
        
        report += f"""
### Business Impact
- ROI is {'positive' if business_impact['roi_percentage'] > 0 else 'negative'} at {business_impact['roi_percentage']:.1f}%
- Focus improvement efforts on countries with highest error rates
- {"Reduce overestimation" if business_impact['overestimation_cost'] > business_impact['underestimation_cost'] else "Reduce underestimation"} costs for maximum impact

### Next Steps
1. {"Immediate model retraining with recent data" if avg_error > 15 else "Continue monitoring with weekly performance reviews"}
2. Investigate {"performance drift causes" if analysis_results['performance_drift']['drift_detected'] else "outlier predictions"}
3. {"Expand to additional countries" if business_impact['roi_percentage'] > 50 else "Optimize existing country predictions"}

---

**Report Generated by**: AAVAIL Post-Production Analysis System  
**Contact**: AI Development Team
"""
        
        return report
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete post-production analysis
        
        Returns:
            dict: Complete analysis results
        """
        print("Starting post-production analysis...")
        
        # Load prediction logs
        df = self.load_prediction_logs()
        print(f"Loaded {len(df)} prediction records")
        
        if df.empty:
            print("No prediction data available for analysis")
            return {}
        
        # Simulate actual revenue (in production, this would be real data)
        df = self.simulate_actual_revenue(df)
        
        # Analyze performance trends
        analysis_results = self.analyze_model_performance_trends(df)
        
        # Calculate business impact
        business_impact = self.calculate_business_impact(df)
        analysis_results['business_impact'] = business_impact
        
        # Generate visualizations
        self.generate_performance_visualizations(df, analysis_results)
        
        # Generate comprehensive report
        report_content = self.generate_comprehensive_report(df, analysis_results, business_impact)
        
        # Save report
        with open('../reports/Post_Production_Analysis_Report.md', 'w') as f:
            f.write(report_content)
        
        print("Post-production analysis completed successfully!")
        print("Report saved to: ../reports/Post_Production_Analysis_Report.md")
        print("Visualizations saved to: ../reports/figures/")
        
        return analysis_results

def main():
    """Main function to run post-production analysis"""
    analyzer = PostProductionAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\nAnalysis Summary:")
        if 'business_impact' in results:
            bi = results['business_impact']
            print(f"ROI: {bi.get('roi_percentage', 0):.1f}%")
            print(f"Average Prediction Error: {bi.get('average_prediction_error', 0):.2f}%")
            print(f"Revenue Accuracy: {bi.get('revenue_accuracy', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
