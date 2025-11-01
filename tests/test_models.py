"""
Unit tests for healthcare models

This module contains unit tests for all the healthcare analytics models
including classification, regression, sentiment analysis, and other components.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import joblib
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestHealthcareModels(unittest.TestCase):
    """Test cases for healthcare models"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        
        # Create sample data for testing
        self.sample_classification_data = pd.DataFrame({
            'age': [65, 45, 78, 32],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'admission_type': ['Emergency', 'Elective', 'Emergency', 'Urgent'],
            'mortality': [1, 0, 1, 0]
        })
        
        self.sample_regression_data = pd.DataFrame({
            'age': [65, 45, 78, 32],
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'admission_type': ['Emergency', 'Elective', 'Emergency', 'Urgent'],
            'los_days': [8.5, 3.2, 15.7, 2.1]
        })
        
        self.sample_text_data = [
            "The hospital staff was very helpful and attentive.",
            "The wait time was too long and the staff was rude.",
            "I had a mixed experience, some things were good but others could be improved."
        ]
    
    def test_classification_model_loading(self):
        """Test classification model loading"""
        model_path = os.path.join(self.base_path, 'models', 'classification', 'mortality_prediction_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                self.assertIsNotNone(model)
                print("✅ Classification model loaded successfully")
            except Exception as e:
                self.fail(f"Failed to load classification model: {e}")
        else:
            print("⚠️ Classification model not found - skipping test")
    
    def test_regression_model_loading(self):
        """Test regression model loading"""
        model_path = os.path.join(self.base_path, 'models', 'regression', 'length_of_stay_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                self.assertIsNotNone(model)
                print("✅ Regression model loaded successfully")
            except Exception as e:
                self.fail(f"Failed to load regression model: {e}")
        else:
            print("⚠️ Regression model not found - skipping test")
    
    def test_sentiment_model_loading(self):
        """Test sentiment analysis model loading"""
        model_path = os.path.join(self.base_path, 'models', 'nlp', 'sentiment_analysis_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                self.assertIsNotNone(model)
                print("✅ Sentiment model loaded successfully")
            except Exception as e:
                self.fail(f"Failed to load sentiment model: {e}")
        else:
            print("⚠️ Sentiment model not found - skipping test")
    
    def test_classification_prediction(self):
        """Test classification model prediction"""
        model_path = os.path.join(self.base_path, 'models', 'classification', 'mortality_prediction_model.pkl')
        scaler_path = os.path.join(self.base_path, 'models', 'classification', 'mortality_prediction_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Create sample input
                sample_input = np.array([[65, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                sample_input_scaled = scaler.transform(sample_input)
                
                prediction = model.predict(sample_input_scaled)
                self.assertIsInstance(prediction, np.ndarray)
                self.assertEqual(len(prediction), 1)
                print("✅ Classification prediction test passed")
                
            except Exception as e:
                self.fail(f"Classification prediction failed: {e}")
        else:
            print("⚠️ Classification model or scaler not found - skipping test")
    
    def test_regression_prediction(self):
        """Test regression model prediction"""
        model_path = os.path.join(self.base_path, 'models', 'regression', 'length_of_stay_model.pkl')
        scaler_path = os.path.join(self.base_path, 'models', 'regression', 'length_of_stay_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Create sample input
                sample_input = np.array([[65, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                sample_input_scaled = scaler.transform(sample_input)
                
                prediction = model.predict(sample_input_scaled)
                self.assertIsInstance(prediction, np.ndarray)
                self.assertEqual(len(prediction), 1)
                self.assertGreater(prediction[0], 0)  # LOS should be positive
                print("✅ Regression prediction test passed")
                
            except Exception as e:
                self.fail(f"Regression prediction failed: {e}")
        else:
            print("⚠️ Regression model or scaler not found - skipping test")
    
    def test_sentiment_prediction(self):
        """Test sentiment analysis prediction"""
        model_path = os.path.join(self.base_path, 'models', 'nlp', 'sentiment_analysis_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                
                # Test positive sentiment
                positive_text = ["The hospital staff was very helpful and attentive."]
                prediction = model.predict(positive_text)
                self.assertIsInstance(prediction, np.ndarray)
                self.assertEqual(len(prediction), 1)
                print("✅ Sentiment prediction test passed")
                
            except Exception as e:
                self.fail(f"Sentiment prediction failed: {e}")
        else:
            print("⚠️ Sentiment model not found - skipping test")
    
    def test_healthcare_chatbot(self):
        """Test healthcare chatbot functionality"""
        try:
            from chatbot.healthcare_chatbot import HealthcareChatbot
            
            chatbot = HealthcareChatbot()
            self.assertIsNotNone(chatbot)
            
            # Test basic chat functionality
            response = chatbot.chat("I have a headache")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            # Test FAQ functionality
            faq_response = chatbot.chat("What are your hours?")
            self.assertIsInstance(faq_response, str)
            
            print("✅ Healthcare chatbot test passed")
            
        except ImportError:
            print("⚠️ Healthcare chatbot not available - skipping test")
        except Exception as e:
            self.fail(f"Healthcare chatbot test failed: {e}")
    
    def test_medical_translator(self):
        """Test medical translator functionality"""
        try:
            from translator.medical_translator import MedicalTranslator
            
            translator = MedicalTranslator()
            self.assertIsNotNone(translator)
            
            # Test supported languages
            languages = translator.get_supported_languages()
            self.assertIsInstance(languages, dict)
            self.assertIn("es", languages)  # Spanish should be supported
            
            # Test medical term translation
            spanish_term = translator.translate_medical_term("headache", "en", "es")
            self.assertIsInstance(spanish_term, str)
            
            print("✅ Medical translator test passed")
            
        except ImportError:
            print("⚠️ Medical translator not available - skipping test")
        except Exception as e:
            self.fail(f"Medical translator test failed: {e}")
    
    def test_data_preprocessing(self):
        """Test data preprocessing functions"""
        # Test basic data cleaning
        data = self.sample_classification_data.copy()
        
        # Test missing value handling
        data_with_missing = data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        
        # Remove missing values
        cleaned_data = data_with_missing.dropna()
        self.assertLess(len(cleaned_data), len(data_with_missing))
        
        # Test categorical encoding
        encoded_data = pd.get_dummies(data, columns=['gender', 'admission_type'], drop_first=True)
        self.assertGreater(len(encoded_data.columns), len(data.columns))
        
        print("✅ Data preprocessing test passed")
    
    def test_model_metrics(self):
        """Test model evaluation metrics"""
        # Test classification metrics
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        
        print("✅ Model metrics test passed")
    
    def test_file_structure(self):
        """Test project file structure"""
        expected_dirs = [
            'src/models/classification',
            'src/models/regression', 
            'src/models/nlp',
            'src/chatbot',
            'src/dashboard',
            'src/translator'
        ]
        
        for dir_path in expected_dirs:
            full_path = os.path.join(self.base_path, dir_path)
            if os.path.exists(full_path):
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"⚠️ Directory missing: {dir_path}")
    
    def tearDown(self):
        """Clean up after tests"""
        pass


class TestAPIIntegration(unittest.TestCase):
    """Test cases for API integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    
    def test_api_script_exists(self):
        """Test that API script exists"""
        api_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'main_api.py')
        self.assertTrue(os.path.exists(api_path), "API script should exist")
        print("✅ API script exists")
    
    def test_dashboard_script_exists(self):
        """Test that dashboard script exists"""
        dashboard_path = os.path.join(self.base_path, 'dashboard', 'healthcare_dashboard.py')
        self.assertTrue(os.path.exists(dashboard_path), "Dashboard script should exist")
        print("✅ Dashboard script exists")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHealthcareModels))
    test_suite.addTest(unittest.makeSuite(TestAPIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
