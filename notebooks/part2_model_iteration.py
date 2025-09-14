#!/usr/bin/env python
"""
AAVAIL Revenue Prediction - Part 2: Model Development & Iteration
Converted from Jupyter notebook to standalone Python script
"""

# Import required libraries
import sys
import os
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import pickle

# Import custom modules
from data_ingestion import load_retail_data
from eda import EDAAnalyzer

def prepare_modeling_data(df, top_countries):
    """Prepare data for modeling"""
    print("\n🔧 Preparing data for modeling...")
    
    # Focus on top countries
    modeling_data = df[df['Country'].isin(top_countries)].copy()
    
    # Calculate revenue
    modeling_data['Revenue'] = modeling_data['Quantity'] * modeling_data['UnitPrice']
    
    # Create time-based features
    modeling_data['Year'] = modeling_data['InvoiceDate'].dt.year
    modeling_data['Month'] = modeling_data['InvoiceDate'].dt.month
    modeling_data['Day'] = modeling_data['InvoiceDate'].dt.day
    modeling_data['DayOfWeek'] = modeling_data['InvoiceDate'].dt.dayofweek
    modeling_data['IsWeekend'] = modeling_data['DayOfWeek'] >= 5
    modeling_data['Quarter'] = modeling_data['InvoiceDate'].dt.quarter
    
    # Aggregate to daily level
    daily_agg = modeling_data.groupby(['Country', modeling_data['InvoiceDate'].dt.date]).agg({
        'Revenue': 'sum',
        'CustomerID': 'nunique',
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'Year': 'first',
        'Month': 'first',
        'Day': 'first',
        'DayOfWeek': 'first',
        'IsWeekend': 'first',
        'Quarter': 'first'
    }).reset_index()
    
    daily_agg.columns = ['Country', 'Date', 'DailyRevenue', 'UniqueCustomers', 
                        'Transactions', 'TotalQuantity', 'Year', 'Month', 'Day', 
                        'DayOfWeek', 'IsWeekend', 'Quarter']
    
    # Encode categorical variables
    le_country = LabelEncoder()
    daily_agg['Country_encoded'] = le_country.fit_transform(daily_agg['Country'])
    
    # Create lag features
    daily_agg = daily_agg.sort_values(['Country', 'Date'])
    daily_agg['Revenue_lag1'] = daily_agg.groupby('Country')['DailyRevenue'].shift(1)
    daily_agg['Revenue_lag7'] = daily_agg.groupby('Country')['DailyRevenue'].shift(7)
    daily_agg['Revenue_ma7'] = daily_agg.groupby('Country')['DailyRevenue'].rolling(window=7).mean().reset_index(0, drop=True)
    
    # Drop rows with NaN values from lag features
    daily_agg = daily_agg.dropna()
    
    print(f"📊 Modeling dataset shape: {daily_agg.shape}")
    print(f"📅 Date range: {daily_agg['Date'].min()} to {daily_agg['Date'].max()}")
    
    return daily_agg, le_country

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n🤖 Training multiple models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        predictions[name] = {
            'train_pred': y_pred_train,
            'test_pred': y_pred_test
        }
        
        print(f"      R² Score: {test_r2:.4f}")
        print(f"      RMSE: {test_rmse:.2f}")
    
    return results, predictions

def evaluate_models(results):
    """Create comprehensive model evaluation"""
    print("\n📊 Model Evaluation Results:")
    print("=" * 80)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            'Model': name,
            'Train_R2': metrics['train_r2'],
            'Test_R2': metrics['test_r2'],
            'Train_RMSE': metrics['train_rmse'],
            'Test_RMSE': metrics['test_rmse'],
            'Train_MAE': metrics['train_mae'],
            'Test_MAE': metrics['test_mae'],
            'Overfitting': metrics['train_r2'] - metrics['test_r2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
    
    print("\n🏆 MODEL PERFORMANCE RANKING:")
    print("-" * 80)
    for i, row in comparison_df.iterrows():
        print(f"{comparison_df.index.get_loc(i)+1}. {row['Model']:<20} R²: {row['Test_R2']:.4f} | RMSE: {row['Test_RMSE']:.2f} | Overfitting: {row['Overfitting']:.4f}")
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_result = results[best_model_name]
    
    print(f"\n🥇 BEST MODEL: {best_model_name}")
    print(f"   Test R² Score: {best_model_result['test_r2']:.4f}")
    print(f"   Test RMSE: {best_model_result['test_rmse']:.2f}")
    print(f"   Test MAE: {best_model_result['test_mae']:.2f}")
    
    return comparison_df, best_model_name, best_model_result

def create_model_visualizations(results, predictions, X_test, y_test):
    """Create comprehensive model visualization"""
    print("\n🎨 Creating model performance visualizations...")
    
    # Model comparison plot
    plt.figure(figsize=(14, 10))
    
    # 1. R² Score comparison
    plt.subplot(2, 2, 1)
    models = list(results.keys())
    test_scores = [results[model]['test_r2'] for model in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = plt.bar(models, test_scores, color=colors, alpha=0.8)
    plt.title('🎯 Model R² Score Comparison', fontweight='bold', fontsize=12)
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    
    # Add score labels
    for bar, score in zip(bars, test_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. RMSE comparison
    plt.subplot(2, 2, 2)
    rmse_scores = [results[model]['test_rmse'] for model in models]
    bars = plt.bar(models, rmse_scores, color='orange', alpha=0.8)
    plt.title('📊 Model RMSE Comparison', fontweight='bold', fontsize=12)
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # 3. Overfitting analysis
    plt.subplot(2, 2, 3)
    train_scores = [results[model]['train_r2'] for model in models]
    x_pos = np.arange(len(models))
    
    plt.plot(x_pos, train_scores, 'o-', label='Train R²', linewidth=2)
    plt.plot(x_pos, test_scores, 's-', label='Test R²', linewidth=2)
    plt.xticks(x_pos, models, rotation=45)
    plt.ylabel('R² Score')
    plt.title('🔍 Overfitting Analysis', fontweight='bold', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Best model predictions vs actual
    best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
    plt.subplot(2, 2, 4)
    y_pred = predictions[best_model]['test_pred']
    
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'🎯 {best_model} Predictions', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Model visualizations saved to ../reports/figures/")

def main():
    print("=" * 80)
    print("AAVAIL REVENUE PREDICTION - PART 2: MODEL DEVELOPMENT & ITERATION")
    print("=" * 80)
    
    # 1. Load processed data from Part 1
    print("\n📥 Loading processed data from Part 1...")
    try:
        # Load the original data
        df = load_retail_data('../data/Online Retail.xlsx')
        
        # Load top countries from processed data
        focused_data = pd.read_csv('../data/processed/focused_data_top10.csv')
        top_countries = focused_data['Country'].unique().tolist()
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🌍 Top countries: {', '.join(top_countries[:5])}...")
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        print("Loading original data and creating top countries list...")
        
        df = load_retail_data('../data/Online Retail.xlsx')
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
        top_countries = country_revenue.head(10).index.tolist()
        
        print(f"✅ Fallback data loaded!")
        print(f"🌍 Top countries: {', '.join(top_countries)}")
    
    # 2. Prepare modeling dataset
    print("\n" + "=" * 60)
    print("2. DATA PREPARATION FOR MODELING")
    print("=" * 60)
    
    modeling_data, le_country = prepare_modeling_data(df, top_countries)
    
    # 3. Feature Engineering
    print("\n🛠️ Feature Engineering...")
    
    feature_columns = [
        'Country_encoded', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter',
        'UniqueCustomers', 'Transactions', 'TotalQuantity',
        'Revenue_lag1', 'Revenue_lag7', 'Revenue_ma7'
    ]
    
    X = modeling_data[feature_columns]
    y = modeling_data['DailyRevenue']
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"🎯 Target: DailyRevenue")
    print(f"📈 Dataset size: {len(X)} samples")
    
    # 4. Train-Test Split
    print("\n✂️ Splitting data...")
    
    # Time-based split (last 20% for testing)
    split_date = modeling_data['Date'].quantile(0.8)
    train_mask = modeling_data['Date'] <= split_date
    
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    
    print(f"📚 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    print(f"📅 Split date: {split_date}")
    
    # 5. Feature Scaling
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Model Training & Evaluation
    print("\n" + "=" * 60)
    print("6. MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    results, predictions = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 7. Model Comparison
    print("\n" + "=" * 60)
    print("7. MODEL COMPARISON & SELECTION")
    print("=" * 60)
    
    comparison_df, best_model_name, best_model_result = evaluate_models(results)
    
    # 8. Create Visualizations
    print("\n" + "=" * 60)
    print("8. PERFORMANCE VISUALIZATIONS")
    print("=" * 60)
    
    create_model_visualizations(results, predictions, X_test_scaled, y_test)
    
    # 9. Save Best Model
    print("\n" + "=" * 60)
    print("9. MODEL PERSISTENCE")
    print("=" * 60)
    
    print(f"\n💾 Saving best model: {best_model_name}...")
    
    # Save model artifacts
    model_data = {
        'best_model_name': best_model_name,
        'best_model': best_model_result['model'],
        'scaler': scaler,
        'label_encoder': le_country,
        'feature_columns': feature_columns,
        'model_metrics': {
            'test_r2': best_model_result['test_r2'],
            'test_rmse': best_model_result['test_rmse'],
            'test_mae': best_model_result['test_mae']
        },
        'model_comparison': comparison_df.to_dict()
    }
    
    os.makedirs('../models', exist_ok=True)
    with open('../models/best_model_assignment02.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model artifacts saved successfully!")
    
    # 10. Feature Importance Analysis
    print("\n" + "=" * 60)
    print("10. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    best_model = best_model_result['model']
    
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance ({best_model_name}):")
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['Importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.xlabel('Importance')
        plt.title(f'🔍 Feature Importance - {best_model_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig('../reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 11. Business Insights
    print("\n" + "=" * 60)
    print("11. BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n=== KEY FINDINGS ===")
    print(f"🥇 Best performing model: {best_model_name}")
    print(f"📊 Prediction accuracy: {best_model_result['test_r2']:.1%}")
    print(f"💰 Average prediction error: €{best_model_result['test_mae']:.2f}")
    
    print("\n=== BUSINESS RECOMMENDATIONS ===")
    print("1. 🎯 DEPLOYMENT STRATEGY:")
    print(f"   • Deploy {best_model_name} for production revenue forecasting")
    print(f"   • Expected accuracy: {best_model_result['test_r2']:.1%}")
    print(f"   • Monitor model performance weekly")
    
    print("\n2. 📈 MODEL IMPROVEMENT:")
    print("   • Collect more granular customer data")
    print("   • Include external factors (holidays, events)")
    print("   • Implement ensemble methods for better accuracy")
    
    print("\n3. 💼 OPERATIONAL USAGE:")
    print("   • Generate daily revenue forecasts")
    print("   • Country-specific predictions for resource allocation")
    print("   • Early warning system for revenue anomalies")
    
    print("\n✅ MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
    print("📋 Ready for Part 3: Production Deployment")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model_result,
        'comparison_df': comparison_df,
        'feature_columns': feature_columns,
        'scaler': scaler
    }

if __name__ == "__main__":
    results = main()
