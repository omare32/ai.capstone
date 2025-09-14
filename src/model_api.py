#!/usr/bin/env python
"""
AAVAIL Revenue Prediction API - Assignment 03
Flask API with train, predict, and logfile endpoints
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib

# Configure logging
import os
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import model functions
sys.path.append(os.path.dirname(__file__))
from data_ingestion import load_retail_data
from model_approaches import TimeSeriesModelApproaches, run_model_comparison

class ModelAPI:
    """
    Production-ready API for AAVAIL Revenue Prediction
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Initialize model storage
        self.models = {}
        self.model_metadata = {}
        self.prediction_logs = []
        
        # Setup routes
        self.setup_routes()
        
        logger.info("Model API initialized successfully")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with API documentation"""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AAVAIL Revenue Prediction API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .method { color: #2196F3; font-weight: bold; }
                    pre { background: #333; color: #fff; padding: 10px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>AAVAIL Revenue Prediction API</h1>
                <p>Production-ready API for monthly revenue forecasting</p>
                
                <div class="endpoint">
                    <h3><span class="method">POST</span> /train</h3>
                    <p>Train models with uploaded data</p>
                    <pre>curl -X POST -F "file=@data.csv" http://localhost:5000/train</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method">POST</span> /predict</h3>
                    <p>Predict next 30 days revenue</p>
                    <pre>curl -X POST -H "Content-Type: application/json" \\
     -d '{"country": "United Kingdom", "date": "2019-08-01"}' \\
     http://localhost:5000/predict</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method">GET</span> /logs</h3>
                    <p>View prediction logs and model performance</p>
                    <pre>curl http://localhost:5000/logs</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method">GET</span> /health</h3>
                    <p>API health check</p>
                    <pre>curl http://localhost:5000/health</pre>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_template)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'models_loaded': len(self.models),
                'api_version': '1.0.0'
            })
        
        @self.app.route('/train', methods=['POST'])
        def train_models():
            """Train models endpoint"""
            try:
                logger.info("Training request received")
                
                # Check if file uploaded or use existing data
                if 'file' in request.files:
                    file = request.files['file']
                    if file.filename != '':
                        # Save uploaded file
                        filepath = os.path.join('data', 'uploaded_data.csv')
                        file.save(filepath)
                        df = pd.read_csv(filepath)
                        df['date'] = pd.to_datetime(df['date'])
                        logger.info(f"Loaded uploaded file: {len(df)} records")
                    else:
                        return jsonify({'error': 'No file uploaded'}), 400
                else:
                    # Use existing processed data
                    try:
                        df = pd.read_csv('data/processed/focused_data_top10.csv')
                        df['date'] = pd.to_datetime(df['date'])
                        logger.info(f"Using existing data: {len(df)} records")
                    except FileNotFoundError:
                        return jsonify({'error': 'No training data available. Please upload data.'}), 400
                
                # Get training parameters
                countries = request.form.get('countries', 'all')
                
                # Train models
                if countries == 'all':
                    results, comparison = run_model_comparison(df, country=None)
                    model_key = 'all_countries'
                else:
                    country_list = countries.split(',')
                    results = {}
                    comparison = pd.DataFrame()
                    
                    for country in country_list:
                        country_results, country_comparison = run_model_comparison(df, country=country.strip())
                        results[country] = country_results
                        model_key = country.strip()
                
                # Select best model
                modeler = TimeSeriesModelApproaches()
                if countries == 'all':
                    best_model_name, best_model_result = modeler.select_best_model(results)
                    self.models[model_key] = best_model_result
                    self.model_metadata[model_key] = {
                        'trained_at': datetime.now().isoformat(),
                        'model_type': best_model_name,
                        'performance': {
                            'mape': best_model_result['mape'],
                            'mae': best_model_result['mae']
                        }
                    }
                
                # Save model to disk
                model_path = f'models/model_{model_key}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[model_key], f)
                
                logger.info(f"Model training completed for {model_key}")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Models trained successfully',
                    'best_model': best_model_name,
                    'performance': {
                        'mape': float(best_model_result['mape']),
                        'mae': float(best_model_result['mae'])
                    },
                    'trained_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                return jsonify({'error': f'Training failed: {str(e)}'}), 500
        
        @self.app.route('/predict', methods=['POST'])
        def predict_revenue():
            """Prediction endpoint"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                country = data.get('country', 'all_countries')
                prediction_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                
                # Validate inputs
                try:
                    pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
                except ValueError:
                    return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
                
                # Determine model key
                model_key = country if country in self.models else 'all_countries'
                
                if model_key not in self.models:
                    return jsonify({'error': f'No trained model available for {country}'}), 404
                
                model_result = self.models[model_key]
                
                # Make prediction
                prediction = float(model_result['forecast_30d_sum'])
                confidence_interval = {
                    'lower': prediction * 0.85,  # Simple confidence interval
                    'upper': prediction * 1.15
                }
                
                # Log prediction
                prediction_log = {
                    'timestamp': datetime.now().isoformat(),
                    'country': country,
                    'prediction_date': prediction_date,
                    'prediction': prediction,
                    'model_type': model_result['model_type'],
                    'model_performance': {
                        'mape': float(model_result['mape']),
                        'mae': float(model_result['mae'])
                    }
                }
                
                self.prediction_logs.append(prediction_log)
                
                # Save log to file
                with open('logs/predictions.json', 'w') as f:
                    json.dump(self.prediction_logs, f, indent=2)
                
                logger.info(f"Prediction made for {country} on {prediction_date}: ${prediction:,.2f}")
                
                return jsonify({
                    'status': 'success',
                    'prediction': {
                        'country': country,
                        'prediction_date': prediction_date,
                        'forecast_period': '30 days',
                        'predicted_revenue': round(prediction, 2),
                        'confidence_interval': confidence_interval,
                        'currency': 'EUR'
                    },
                    'model_info': {
                        'model_type': model_result['model_type'],
                        'mape': round(float(model_result['mape']) * 100, 2),
                        'mae': round(float(model_result['mae']), 2)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        @self.app.route('/logs', methods=['GET'])
        def get_logs():
            """Get prediction logs and performance metrics"""
            try:
                # Get query parameters
                limit = request.args.get('limit', 100, type=int)
                country = request.args.get('country', None)
                
                # Filter logs
                filtered_logs = self.prediction_logs
                if country:
                    filtered_logs = [log for log in filtered_logs if log['country'] == country]
                
                # Limit results
                filtered_logs = filtered_logs[-limit:]
                
                # Calculate summary statistics
                if filtered_logs:
                    predictions = [log['prediction'] for log in filtered_logs]
                    summary_stats = {
                        'total_predictions': len(filtered_logs),
                        'avg_prediction': np.mean(predictions),
                        'min_prediction': np.min(predictions),
                        'max_prediction': np.max(predictions),
                        'latest_prediction': filtered_logs[-1] if filtered_logs else None
                    }
                else:
                    summary_stats = {
                        'total_predictions': 0,
                        'message': 'No predictions found'
                    }
                
                return jsonify({
                    'status': 'success',
                    'logs': filtered_logs,
                    'summary': summary_stats,
                    'models_available': list(self.models.keys()),
                    'api_stats': {
                        'uptime': 'Available',
                        'total_models': len(self.models),
                        'total_predictions': len(self.prediction_logs)
                    }
                })
                
            except Exception as e:
                logger.error(f"Logs retrieval failed: {str(e)}")
                return jsonify({'error': f'Failed to retrieve logs: {str(e)}'}), 500
        
        @self.app.route('/retrain', methods=['POST'])
        def retrain_models():
            """Retrain models (for production deployment)"""
            try:
                logger.info("Retraining request received")
                
                # This would typically be called on a schedule
                # For now, we'll just call the train endpoint logic
                return train_models()
                
            except Exception as e:
                logger.error(f"Retraining failed: {str(e)}")
                return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
    
    def load_existing_models(self):
        """Load pre-trained models if available"""
        try:
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_path = os.path.join('models', model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_key = model_file.replace('model_', '').replace('.pkl', '')
                self.models[model_key] = model_data
                
                logger.info(f"Loaded existing model: {model_key}")
            
        except Exception as e:
            logger.warning(f"Could not load existing models: {str(e)}")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.load_existing_models()
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create API instance
api = ModelAPI()

if __name__ == '__main__':
    # Run the API
    api.run(debug=True)
