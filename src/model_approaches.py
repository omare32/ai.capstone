#!/usr/bin/env python
"""
Model Approaches for AAVAIL Revenue Prediction - Assignment 02
Time-series forecasting models to predict next 30 days revenue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Traditional time-series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be skipped.")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesModelApproaches:
    """
    Comprehensive time-series modeling approaches for revenue prediction
    """
    
    def __init__(self, target_days: int = 30):
        """
        Initialize modeling approaches
        
        Args:
            target_days: Number of days to predict (default 30)
        """
        self.target_days = target_days
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def prepare_time_series_data(self, df: pd.DataFrame, country: str = None) -> pd.DataFrame:
        """
        Prepare data for time-series modeling
        
        Args:
            df: Input DataFrame with transaction data
            country: Specific country to filter (None for all countries)
            
        Returns:
            DataFrame: Daily aggregated time-series data
        """
        # Filter by country if specified
        if country:
            df_filtered = df[df['country'] == country].copy()
        else:
            df_filtered = df.copy()
        
        # Create daily aggregates
        daily_data = df_filtered.groupby('date').agg({
            'price': 'sum',
            'customer_id': 'nunique',
            'invoice': 'nunique',
            'times_viewed': 'mean'
        }).reset_index()
        
        daily_data.columns = ['date', 'daily_revenue', 'unique_customers', 'unique_invoices', 'avg_views']
        
        # Create date range to fill missing dates
        date_range = pd.date_range(start=daily_data['date'].min(), 
                                 end=daily_data['date'].max(), 
                                 freq='D')
        
        # Reindex to include all dates
        daily_data = daily_data.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_data.columns = ['date', 'daily_revenue', 'unique_customers', 'unique_invoices', 'avg_views']
        
        # Add time features
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['day_of_month'] = daily_data['date'].dt.day
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['quarter'] = daily_data['date'].dt.quarter
        daily_data['year'] = daily_data['date'].dt.year
        daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate rolling features
        daily_data['revenue_7d_avg'] = daily_data['daily_revenue'].rolling(window=7, min_periods=1).mean()
        daily_data['revenue_30d_avg'] = daily_data['daily_revenue'].rolling(window=30, min_periods=1).mean()
        daily_data['revenue_7d_std'] = daily_data['daily_revenue'].rolling(window=7, min_periods=1).std()
        
        return daily_data
    
    def create_supervised_features(self, ts_data: pd.DataFrame, lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create supervised learning features from time series data
        
        Args:
            ts_data: Time series DataFrame
            lookback_days: Number of historical days to use as features
            
        Returns:
            tuple: (X features, y targets) for next 30 days revenue
        """
        revenue_series = ts_data['daily_revenue'].values
        
        X, y = [], []
        
        # Create sliding windows
        for i in range(lookback_days, len(revenue_series) - self.target_days + 1):
            # Features: lookback_days of historical data
            X.append(revenue_series[i-lookback_days:i])
            # Target: sum of next target_days
            y.append(np.sum(revenue_series[i:i+self.target_days]))
        
        return np.array(X), np.array(y)
    
    def approach_1_arima(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Approach 1: ARIMA Time Series Model
        """
        logger.info("Training ARIMA model...")
        
        revenue_series = ts_data.set_index('date')['daily_revenue']
        
        # Split data for validation
        train_size = int(len(revenue_series) * 0.8)
        train_data = revenue_series[:train_size]
        test_data = revenue_series[train_size:]
        
        try:
            # Fit ARIMA model (auto-determine parameters)
            model = ARIMA(train_data, order=(2, 1, 2))  # Start with common parameters
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, predictions)
            mse = mean_squared_error(test_data, predictions)
            mape = mean_absolute_percentage_error(test_data, predictions)
            
            # Make 30-day forecast
            forecast_30d = fitted_model.forecast(steps=self.target_days)
            forecast_30d_sum = np.sum(forecast_30d)
            
            result = {
                'model': fitted_model,
                'model_type': 'ARIMA',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'forecast_30d_sum': forecast_30d_sum,
                'forecast_30d_daily': forecast_30d,
                'train_size': train_size,
                'test_size': len(test_data)
            }
            
            logger.info(f"ARIMA - MAE: {mae:.2f}, MAPE: {mape:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"ARIMA model failed: {e}")
            return {'model_type': 'ARIMA', 'error': str(e)}
    
    def approach_2_exponential_smoothing(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Approach 2: Exponential Smoothing (Holt-Winters)
        """
        logger.info("Training Exponential Smoothing model...")
        
        revenue_series = ts_data.set_index('date')['daily_revenue']
        
        # Split data
        train_size = int(len(revenue_series) * 0.8)
        train_data = revenue_series[:train_size]
        test_data = revenue_series[train_size:]
        
        try:
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(train_data, 
                                       trend='add', 
                                       seasonal='add', 
                                       seasonal_periods=7)  # Weekly seasonality
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, predictions)
            mse = mean_squared_error(test_data, predictions)
            mape = mean_absolute_percentage_error(test_data, predictions)
            
            # Make 30-day forecast
            forecast_30d = fitted_model.forecast(steps=self.target_days)
            forecast_30d_sum = np.sum(forecast_30d)
            
            result = {
                'model': fitted_model,
                'model_type': 'Exponential Smoothing',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'forecast_30d_sum': forecast_30d_sum,
                'forecast_30d_daily': forecast_30d,
                'train_size': train_size,
                'test_size': len(test_data)
            }
            
            logger.info(f"Exponential Smoothing - MAE: {mae:.2f}, MAPE: {mape:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Exponential Smoothing model failed: {e}")
            return {'model_type': 'Exponential Smoothing', 'error': str(e)}
    
    def approach_3_random_forest(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Approach 3: Random Forest with engineered features
        """
        logger.info("Training Random Forest model...")
        
        try:
            # Create supervised learning dataset
            X, y = self.create_supervised_features(ts_data, lookback_days=30)
            
            if len(X) == 0:
                return {'model_type': 'Random Forest', 'error': 'Insufficient data for feature creation'}
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # Make future prediction (use last 30 days)
            last_30_days = ts_data['daily_revenue'].tail(30).values.reshape(1, -1)
            last_30_days_scaled = scaler.transform(last_30_days)
            forecast_30d_sum = model.predict(last_30_days_scaled)[0]
            
            result = {
                'model': model,
                'scaler': scaler,
                'model_type': 'Random Forest',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'forecast_30d_sum': forecast_30d_sum,
                'train_size': train_size,
                'test_size': len(X_test),
                'feature_importance': model.feature_importances_
            }
            
            logger.info(f"Random Forest - MAE: {mae:.2f}, MAPE: {mape:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Random Forest model failed: {e}")
            return {'model_type': 'Random Forest', 'error': str(e)}
    
    def approach_4_gradient_boosting(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Approach 4: Gradient Boosting Regressor
        """
        logger.info("Training Gradient Boosting model...")
        
        try:
            # Create supervised learning dataset
            X, y = self.create_supervised_features(ts_data, lookback_days=30)
            
            if len(X) == 0:
                return {'model_type': 'Gradient Boosting', 'error': 'Insufficient data for feature creation'}
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # Make future prediction
            last_30_days = ts_data['daily_revenue'].tail(30).values.reshape(1, -1)
            last_30_days_scaled = scaler.transform(last_30_days)
            forecast_30d_sum = model.predict(last_30_days_scaled)[0]
            
            result = {
                'model': model,
                'scaler': scaler,
                'model_type': 'Gradient Boosting',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'forecast_30d_sum': forecast_30d_sum,
                'train_size': train_size,
                'test_size': len(X_test),
                'feature_importance': model.feature_importances_
            }
            
            logger.info(f"Gradient Boosting - MAE: {mae:.2f}, MAPE: {mape:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Gradient Boosting model failed: {e}")
            return {'model_type': 'Gradient Boosting', 'error': str(e)}
    
    def approach_5_lstm(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Approach 5: LSTM Neural Network (if TensorFlow available)
        """
        if not TENSORFLOW_AVAILABLE:
            return {'model_type': 'LSTM', 'error': 'TensorFlow not available'}
        
        logger.info("Training LSTM model...")
        
        try:
            # Create supervised learning dataset
            X, y = self.create_supervised_features(ts_data, lookback_days=60)  # Longer lookback for LSTM
            
            if len(X) < 100:  # Need sufficient data for LSTM
                return {'model_type': 'LSTM', 'error': 'Insufficient data for LSTM training'}
            
            # Reshape for LSTM (samples, timesteps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            history = model.fit(X_train_scaled, y_train_scaled, 
                              epochs=50, batch_size=32, verbose=0,
                              validation_split=0.2)
            
            # Make predictions
            predictions_scaled = model.predict(X_test_scaled, verbose=0)
            predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # Make future prediction
            last_60_days = ts_data['daily_revenue'].tail(60).values.reshape(1, 60, 1)
            last_60_days_scaled = scaler_X.transform(last_60_days.reshape(-1, 60)).reshape(1, 60, 1)
            forecast_scaled = model.predict(last_60_days_scaled, verbose=0)
            forecast_30d_sum = scaler_y.inverse_transform(forecast_scaled)[0][0]
            
            result = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'model_type': 'LSTM',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'forecast_30d_sum': forecast_30d_sum,
                'train_size': train_size,
                'test_size': len(X_test),
                'training_history': history.history
            }
            
            logger.info(f"LSTM - MAE: {mae:.2f}, MAPE: {mape:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"LSTM model failed: {e}")
            return {'model_type': 'LSTM', 'error': str(e)}
    
    def compare_all_approaches(self, ts_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compare all modeling approaches
        
        Args:
            ts_data: Time series DataFrame
            
        Returns:
            dict: Results from all approaches
        """
        logger.info("Comparing all modeling approaches...")
        
        approaches = [
            self.approach_1_arima,
            self.approach_2_exponential_smoothing,
            self.approach_3_random_forest,
            self.approach_4_gradient_boosting,
            self.approach_5_lstm
        ]
        
        results = {}
        
        for approach in approaches:
            try:
                result = approach(ts_data)
                results[result['model_type']] = result
            except Exception as e:
                logger.error(f"Failed to run approach: {e}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Select the best model based on MAPE score
        
        Args:
            results: Results from all approaches
            
        Returns:
            tuple: (best_model_name, best_model_result)
        """
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            raise Exception("No valid models found")
        
        # Sort by MAPE (lower is better)
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mape'])
        
        best_model_name = sorted_results[0][0]
        best_model_result = sorted_results[0][1]
        
        logger.info(f"Best model: {best_model_name} with MAPE: {best_model_result['mape']:.2%}")
        
        return best_model_name, best_model_result
    
    def generate_model_comparison_report(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Generate comparison report of all models
        
        Args:
            results: Results from all approaches
            
        Returns:
            DataFrame: Model comparison report
        """
        comparison_data = []
        
        for model_name, result in results.items():
            if 'error' in result:
                comparison_data.append({
                    'Model': model_name,
                    'Status': 'Failed',
                    'Error': result['error'],
                    'MAE': None,
                    'MAPE': None,
                    'Forecast_30d': None
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Status': 'Success',
                    'Error': None,
                    'MAE': result['mae'],
                    'MAPE': result['mape'],
                    'Forecast_30d': result['forecast_30d_sum']
                })
        
        return pd.DataFrame(comparison_data)

def run_model_comparison(df: pd.DataFrame, country: str = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Main function to run model comparison
    
    Args:
        df: Transaction DataFrame
        country: Country to analyze (None for all)
        
    Returns:
        tuple: (results_dict, comparison_dataframe)
    """
    # Initialize modeling framework
    modeler = TimeSeriesModelApproaches(target_days=30)
    
    # Prepare time series data
    ts_data = modeler.prepare_time_series_data(df, country)
    
    # Compare all approaches
    results = modeler.compare_all_approaches(ts_data)
    
    # Generate comparison report
    comparison_df = modeler.generate_model_comparison_report(results)
    
    return results, comparison_df
