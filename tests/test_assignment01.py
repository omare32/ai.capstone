#!/usr/bin/env python
"""
Test Suite for Assignment 01 - Data Investigation
Tests all components of the AAVAIL Revenue Prediction project
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion import DataIngestion, load_retail_data
from eda import EDAAnalyzer, perform_eda

class TestDataIngestion(unittest.TestCase):
    """Test cases for data ingestion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = "../ai-workflow-capstone-master/cs-train"
        
    def test_data_ingestion_initialization(self):
        """Test DataIngestion class initialization"""
        ingestion = DataIngestion(self.test_data_dir)
        self.assertEqual(ingestion.data_directory, self.test_data_dir)
        self.assertIsInstance(ingestion.required_columns, list)
        self.assertGreater(len(ingestion.required_columns), 0)
    
    def test_column_standardization(self):
        """Test column name standardization"""
        ingestion = DataIngestion(self.test_data_dir)
        
        # Create test dataframe with non-standard column names
        test_df = pd.DataFrame({
            'StreamID': [1, 2, 3],
            'TimesViewed': [10, 20, 30],
            'CustomerID': ['A', 'B', 'C']
        })
        
        standardized_df = ingestion.standardize_column_names(test_df)
        
        self.assertIn('stream_id', standardized_df.columns)
        self.assertIn('times_viewed', standardized_df.columns)
        self.assertIn('customer_id', standardized_df.columns)
    
    def test_date_feature_creation(self):
        """Test date feature engineering"""
        ingestion = DataIngestion(self.test_data_dir)
        
        test_df = pd.DataFrame({
            'year': [2018, 2019, 2020],
            'month': [1, 6, 12],
            'day': [15, 20, 25]
        })
        
        df_with_dates = ingestion.create_date_features(test_df)
        
        self.assertIn('date', df_with_dates.columns)
        self.assertIn('month_year', df_with_dates.columns)
        self.assertIn('quarter', df_with_dates.columns)
        self.assertIn('day_of_week', df_with_dates.columns)
        self.assertIn('is_weekend', df_with_dates.columns)

class TestEDAAnalyzer(unittest.TestCase):
    """Test cases for EDA functionality"""
    
    def setUp(self):
        """Set up test fixtures with sample data"""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'country': np.random.choice(['UK', 'Germany', 'France'], 100),
            'customer_id': np.random.choice(range(1, 21), 100),
            'price': np.random.uniform(10, 500, 100),
            'date': pd.date_range('2018-01-01', periods=100, freq='D'),
            'times_viewed': np.random.randint(1, 20, 100),
            'invoice': np.random.choice(range(1000, 2000), 100),
            'day': np.random.randint(1, 32, 100),
            'month': np.random.randint(1, 13, 100),
            'year': np.random.choice([2018, 2019], 100),
            'is_weekend': np.random.choice([True, False], 100)
        })
        
        # Add required derived columns
        self.sample_data['month_year'] = self.sample_data['date'].dt.to_period('M')
        self.sample_data['quarter'] = self.sample_data['date'].dt.quarter
        self.sample_data['day_of_week'] = self.sample_data['date'].dt.dayofweek
        
        self.analyzer = EDAAnalyzer(self.sample_data)
    
    def test_eda_initialization(self):
        """Test EDA analyzer initialization"""
        self.assertIsInstance(self.analyzer.df, pd.DataFrame)
        self.assertEqual(len(self.analyzer.df), 100)
    
    def test_data_summary_generation(self):
        """Test data summary generation"""
        summary = self.analyzer.generate_data_summary()
        
        self.assertIn('basic_info', summary)
        self.assertIn('revenue_stats', summary)
        self.assertIn('missing_data', summary)
        
        # Check basic info structure
        basic_info = summary['basic_info']
        self.assertIn('total_records', basic_info)
        self.assertIn('total_revenue', basic_info)
        self.assertIn('unique_customers', basic_info)
        self.assertEqual(basic_info['total_records'], 100)
    
    def test_country_analysis(self):
        """Test country revenue analysis"""
        country_analysis = self.analyzer.analyze_revenue_by_country(top_n=3)
        
        self.assertIsInstance(country_analysis, pd.DataFrame)
        self.assertLessEqual(len(country_analysis), 3)
        self.assertIn('total_revenue', country_analysis.columns)
        self.assertIn('revenue_percentage', country_analysis.columns)
        
        # Check that revenue percentages sum to 100 or less
        self.assertLessEqual(country_analysis['revenue_percentage'].sum(), 100)
    
    def test_temporal_analysis(self):
        """Test temporal pattern analysis"""
        temporal_results = self.analyzer.analyze_temporal_patterns()
        
        self.assertIn('monthly_trends', temporal_results)
        self.assertIn('daily_patterns', temporal_results)
        self.assertIn('quarterly_patterns', temporal_results)
        
        # Check monthly trends structure
        monthly_trends = temporal_results['monthly_trends']
        self.assertIn('total_revenue', monthly_trends.columns)
        self.assertIn('transaction_count', monthly_trends.columns)
    
    def test_customer_segmentation(self):
        """Test customer analysis and segmentation"""
        customer_analysis = self.analyzer.analyze_customer_segments()
        
        self.assertIsInstance(customer_analysis, pd.DataFrame)
        self.assertIn('total_spent', customer_analysis.columns)
        self.assertIn('customer_value', customer_analysis.columns)
        
        # Check that customer value segments are created
        value_segments = customer_analysis['customer_value'].unique()
        self.assertGreater(len(value_segments), 0)
    
    def test_hypothesis_generation(self):
        """Test hypothesis generation"""
        hypotheses = self.analyzer.generate_hypotheses()
        
        self.assertIsInstance(hypotheses, list)
        self.assertGreater(len(hypotheses), 0)
        
        # Check that all hypotheses start with H and a number
        for hypothesis in hypotheses:
            self.assertTrue(hypothesis.startswith('H'))
    
    def test_hypothesis_testing(self):
        """Test hypothesis testing functionality"""
        test_results = self.analyzer.test_hypotheses()
        
        self.assertIn('pareto_principle', test_results)
        self.assertIn('weekend_vs_weekday', test_results)
        
        # Check pareto principle test structure
        pareto_test = test_results['pareto_principle']
        self.assertIn('top_10_percentage', pareto_test)
        self.assertIn('passes_80_20_rule', pareto_test)
    
    def test_top_countries_identification(self):
        """Test top countries identification"""
        top_countries = self.analyzer.identify_top_countries(n=3)
        
        self.assertIsInstance(top_countries, list)
        self.assertLessEqual(len(top_countries), 3)
        
        # All returned values should be strings (country names)
        for country in top_countries:
            self.assertIsInstance(country, str)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow with simulated data"""
        # Create comprehensive simulated dataset
        np.random.seed(42)
        
        countries = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'Ireland']
        customers = range(1, 101)
        
        # Generate 1000 transactions over 12 months
        data = []
        for i in range(1000):
            data.append({
                'country': np.random.choice(countries),
                'customer_id': np.random.choice(customers),
                'price': np.random.uniform(10, 1000),
                'year': 2018,
                'month': np.random.randint(1, 13),
                'day': np.random.randint(1, 29),
                'times_viewed': np.random.randint(1, 50),
                'invoice': f"INV{i:04d}",
                'stream_id': np.random.randint(1, 100)
            })
        
        df = pd.DataFrame(data)
        
        # Add date features
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['month_year'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Test EDA workflow
        analyzer = EDAAnalyzer(df)
        
        # Test all major functions
        summary = analyzer.generate_data_summary()
        country_analysis = analyzer.analyze_revenue_by_country()
        temporal_analysis = analyzer.analyze_temporal_patterns()
        customer_analysis = analyzer.analyze_customer_segments()
        hypotheses = analyzer.generate_hypotheses()
        hypothesis_tests = analyzer.test_hypotheses()
        top_countries = analyzer.identify_top_countries()
        
        # Verify results
        self.assertEqual(summary['basic_info']['total_records'], 1000)
        self.assertGreater(len(country_analysis), 0)
        self.assertIn('monthly_trends', temporal_analysis)
        self.assertGreater(len(customer_analysis), 0)
        self.assertEqual(len(hypotheses), 8)  # We expect 8 hypotheses
        self.assertIn('pareto_principle', hypothesis_tests)
        self.assertLessEqual(len(top_countries), 10)

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("=" * 60)
    print("AAVAIL Revenue Prediction - Assignment 01 Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestEDAAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Assignment 01 is ready for submission!")
    else:
        print("\n❌ Some tests failed - Review and fix issues before submission")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
