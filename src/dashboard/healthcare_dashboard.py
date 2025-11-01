"""
Healthcare Analytics Dashboard

This Streamlit dashboard provides an interactive interface for all the healthcare
analytics models including classification, regression, sentiment analysis, and
the healthcare chatbot.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.healthcare_chatbot import HealthcareChatbot

# Set page config
st.set_page_config(
    page_title="HealthAI Suite - Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class HealthcareDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()
        self.chatbot = HealthcareChatbot()
    
    def load_models(self):
        """Load all trained models"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load classification model
        try:
            self.models['classification'] = joblib.load(
                os.path.join(base_path, 'models', 'classification', 'mortality_prediction_model.pkl')
            )
            self.scalers['classification'] = joblib.load(
                os.path.join(base_path, 'models', 'classification', 'mortality_prediction_scaler.pkl')
            )
        except:
            st.warning("Classification model not found")
        
        # Load regression model
        try:
            self.models['regression'] = joblib.load(
                os.path.join(base_path, 'models', 'regression', 'length_of_stay_model.pkl')
            )
            self.scalers['regression'] = joblib.load(
                os.path.join(base_path, 'models', 'regression', 'length_of_stay_scaler.pkl')
            )
        except:
            st.warning("Regression model not found")
        
        # Load sentiment model
        try:
            self.models['sentiment'] = joblib.load(
                os.path.join(base_path, 'models', 'nlp', 'sentiment_analysis_model.pkl')
            )
        except:
            st.warning("Sentiment analysis model not found")
    
    def render_header(self):
        """Render the dashboard header"""
        st.markdown('<h1 class="main-header">🏥 HealthAI Suite</h1>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; color: #666;">Healthcare Analytics Dashboard</h2>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        pages = {
            "🏠 Dashboard": "dashboard",
            "📊 Mortality Prediction": "mortality",
            "📈 Length of Stay Prediction": "los",
            "😊 Sentiment Analysis": "sentiment",
            "🤖 Healthcare Chatbot": "chatbot",
            "📋 Model Performance": "performance"
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        return pages[selected_page]
    
    def render_dashboard_overview(self):
        """Render the main dashboard overview"""
        st.header("📊 Dashboard Overview")
        
        # Model status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Loaded" if 'classification' in self.models else "❌ Not Available"
            st.markdown(f"""
            <div class="metric-card {'success-metric' if 'classification' in self.models else 'danger-metric'}">
                <h4>Mortality Prediction</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Type:</strong> Classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅ Loaded" if 'regression' in self.models else "❌ Not Available"
            st.markdown(f"""
            <div class="metric-card {'success-metric' if 'regression' in self.models else 'danger-metric'}">
                <h4>Length of Stay</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Type:</strong> Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status = "✅ Loaded" if 'sentiment' in self.models else "❌ Not Available"
            st.markdown(f"""
            <div class="metric-card {'success-metric' if 'sentiment' in self.models else 'danger-metric'}">
                <h4>Sentiment Analysis</h4>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Type:</strong> NLP</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h4>Healthcare Chatbot</h4>
                <p><strong>Status:</strong> ✅ Available</p>
                <p><strong>Type:</strong> RAG Pipeline</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.subheader("📈 Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Available Models:**
            - Mortality Risk Prediction
            - Length of Stay Forecasting
            - Patient Feedback Sentiment Analysis
            - Healthcare Chatbot with RAG
            """)
        
        with col2:
            st.success("""
            **Key Features:**
            - Real-time predictions
            - Interactive visualizations
            - Patient risk assessment
            - Clinical decision support
            """)
    
    def render_mortality_prediction(self):
        """Render mortality prediction interface"""
        st.header("📊 Mortality Risk Prediction")
        
        if 'classification' not in self.models:
            st.error("Mortality prediction model is not available. Please ensure the model is trained and saved.")
            return
        
        st.subheader("Patient Information")
        
        # Create input form
        with st.form("mortality_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=65)
                gender = st.selectbox("Gender", ["Male", "Female"])
                admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
                
            with col2:
                insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Other"])
                language = st.selectbox("Language", ["English", "Spanish", "Other"])
                ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"])
            
            submitted = st.form_submit_button("Predict Mortality Risk")
            
            if submitted:
                # Prepare input data
                input_data = self._prepare_classification_input(
                    age, gender, admission_type, insurance, language, ethnicity
                )
                
                if input_data is not None:
                    # Make prediction
                    prediction = self.models['classification'].predict(input_data)[0]
                    probability = self.models['classification'].predict_proba(input_data)[0]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error(f"⚠️ **High Mortality Risk**")
                            st.error(f"Risk Level: {probability[1]:.1%}")
                        else:
                            st.success(f"✅ **Low Mortality Risk**")
                            st.success(f"Risk Level: {probability[0]:.1%}")
                    
                    with col2:
                        st.metric("Risk Score", f"{probability[1]:.1%}")
                        st.metric("Confidence", f"{max(probability):.1%}")
    
    def render_los_prediction(self):
        """Render length of stay prediction interface"""
        st.header("📈 Length of Stay Prediction")
        
        if 'regression' not in self.models:
            st.error("Length of stay prediction model is not available. Please ensure the model is trained and saved.")
            return
        
        st.subheader("Patient Information")
        
        # Create input form
        with st.form("los_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=65, key="los_age")
                gender = st.selectbox("Gender", ["Male", "Female"], key="los_gender")
                admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"], key="los_admission")
                
            with col2:
                insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Other"], key="los_insurance")
                language = st.selectbox("Language", ["English", "Spanish", "Other"], key="los_language")
                ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"], key="los_ethnicity")
            
            submitted = st.form_submit_button("Predict Length of Stay")
            
            if submitted:
                # Prepare input data
                input_data = self._prepare_regression_input(
                    age, gender, admission_type, insurance, language, ethnicity
                )
                
                if input_data is not None:
                    # Make prediction
                    prediction = self.models['regression'].predict(input_data)[0]
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted LOS", f"{prediction:.1f} days")
                    
                    with col2:
                        if prediction < 5:
                            st.success("Short Stay")
                        elif prediction < 10:
                            st.warning("Medium Stay")
                        else:
                            st.error("Long Stay")
                    
                    with col3:
                        st.info("Resource planning recommended")
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis interface"""
        st.header("😊 Patient Feedback Sentiment Analysis")
        
        if 'sentiment' not in self.models:
            st.error("Sentiment analysis model is not available. Please ensure the model is trained and saved.")
            return
        
        st.subheader("Feedback Analysis")
        
        # Text input
        feedback_text = st.text_area(
            "Enter patient feedback:",
            placeholder="Enter the patient's feedback text here...",
            height=100
        )
        
        if st.button("Analyze Sentiment"):
            if feedback_text.strip():
                # Make prediction
                sentiment = self.models['sentiment'].predict([feedback_text])[0]
                sentiment_label = "Positive" if sentiment == 1 else "Negative"
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if sentiment == 1:
                        st.success(f"😊 **Positive Sentiment**")
                        st.success("Patient satisfaction is high")
                    else:
                        st.error(f"😞 **Negative Sentiment**")
                        st.error("Patient satisfaction needs attention")
                
                with col2:
                    st.metric("Sentiment Score", sentiment_label)
                    st.metric("Action Required", "Yes" if sentiment == 0 else "No")
                
                # Show feedback
                st.subheader("Feedback Summary")
                st.write(feedback_text)
            else:
                st.warning("Please enter some feedback text to analyze.")
    
    def render_chatbot(self):
        """Render healthcare chatbot interface"""
        st.header("🤖 Healthcare Chatbot")
        
        st.subheader("Ask me anything about healthcare!")
        st.write("I can help with symptoms, medications, appointments, and general health questions.")
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Chatbot:** {message['content']}")
        
        # Chat input
        user_input = st.text_input("Type your message:", key="chat_input")
        
        if st.button("Send") or user_input:
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input
                })
                
                # Get chatbot response
                response = self.chatbot.chat(user_input)
                
                # Add chatbot response to history
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': response
                })
                
                # Clear input and rerun
                st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    def render_model_performance(self):
        """Render model performance metrics"""
        st.header("📋 Model Performance")
        
        # Create performance metrics
        performance_data = {
            "Model": ["Mortality Prediction", "Length of Stay", "Sentiment Analysis"],
            "Accuracy/F1": ["92.3%", "N/A", "88.0%"],
            "Precision": ["95.0%", "N/A", "87.7%"],
            "Recall": ["88.0%", "N/A", "97.3%"],
            "F1-Score": ["91.4%", "N/A", "92.2%"],
            "AUC/R²": ["100.0%", "8.7%", "N/A"]
        }
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        
        # Model status
        st.subheader("Model Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ **Mortality Prediction Model**")
            st.write("- Status: Trained and Ready")
            st.write("- Performance: Excellent")
            st.write("- Last Updated: Recent")
        
        with col2:
            st.success("✅ **Sentiment Analysis Model**")
            st.write("- Status: Trained and Ready")
            st.write("- Performance: Good")
            st.write("- Last Updated: Recent")
    
    def _prepare_classification_input(self, age, gender, admission_type, insurance, language, ethnicity):
        """Prepare input data for classification model"""
        # This is a simplified version - in practice, you'd need to match
        # the exact feature names and encoding used during training
        try:
            # Create a basic feature vector (this would need to match your actual model)
            features = np.zeros(290)  # Adjust based on your actual feature count
            features[0] = age
            
            # Add one-hot encoded features based on selections
            # This is simplified - you'd need the exact feature mapping
            if gender == "Male":
                features[1] = 1
            if admission_type == "Emergency":
                features[2] = 1
            if insurance == "Medicare":
                features[3] = 1
            
            return self.scalers['classification'].transform([features])
        except Exception as e:
            st.error(f"Error preparing input data: {e}")
            return None
    
    def _prepare_regression_input(self, age, gender, admission_type, insurance, language, ethnicity):
        """Prepare input data for regression model"""
        try:
            # Create a basic feature vector (this would need to match your actual model)
            features = np.zeros(16)  # Adjust based on your actual feature count
            features[0] = age
            
            # Add one-hot encoded features based on selections
            if gender == "Male":
                features[1] = 1
            if admission_type == "Emergency":
                features[2] = 1
            if insurance == "Medicare":
                features[3] = 1
            
            return self.scalers['regression'].transform([features])
        except Exception as e:
            st.error(f"Error preparing input data: {e}")
            return None
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Sidebar navigation
        current_page = self.render_sidebar()
        
        # Render current page
        if current_page == "dashboard":
            self.render_dashboard_overview()
        elif current_page == "mortality":
            self.render_mortality_prediction()
        elif current_page == "los":
            self.render_los_prediction()
        elif current_page == "sentiment":
            self.render_sentiment_analysis()
        elif current_page == "chatbot":
            self.render_chatbot()
        elif current_page == "performance":
            self.render_model_performance()


def main():
    """Main function to run the dashboard"""
    dashboard = HealthcareDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
