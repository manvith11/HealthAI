# HealthAI Suite — Intelligent Analytics for Patient Care

An end-to-end AI/ML system that analyzes patient health data to improve clinical decision support, patient engagement, and operational efficiency.

## Project Overview

HealthAI Suite integrates multiple AI paradigms to:
1. Predict outcomes (regression)
2. Classify disease risk categories
3. Discover patient subgroups (clustering)
4. Mine medical associations
5. Build and compare deep learning models (NN, CNN, RNN, LSTM)
6. Leverage pretrained models (BioBERT, ClinicalBERT)
7. Develop a healthcare chatbot for patient queries
8. Build a translator for multilingual medical communication
9. Perform sentiment analysis on patient feedback

## Business Use Cases

- **Risk Stratification**: Early detection of diseases like diabetes, heart disease, cancer staging
- **Length of Stay Prediction**: Forecast patient hospitalization duration for resource planning
- **Patient Segmentation**: Group patients into cohorts (chronic vs. acute, lifestyle-driven vs. genetic-risk)
- **Medical Associations**: Discover patterns like "(high BMI ∧ hypertension) ⇒ increased risk of diabetes"
- **Imaging Diagnostics**: Automate radiology analysis for X-rays, CT, or MRI
- **Sequence Modeling**: Track patient vitals over time to forecast deterioration or readmission
- **Pretrained Models**: Use BioBERT/ClinicalBERT for clinical notes, discharge summaries, drug side effects
- **Healthcare Chatbot**: Patient triage bot for symptoms, appointment scheduling, and FAQs
- **Translator**: Bridge doctor–patient communication in regional languages
- **Sentiment Analysis**: Capture patient experience from feedback/reviews to improve hospital services

## Project Structure

```
HealthAI-Suite/
├── data/                      # Data storage and preprocessing
├── models/                    # Model training and evaluation
│   ├── classification/        # Disease risk prediction models
│   ├── regression/            # Length of stay prediction models
│   ├── clustering/            # Patient segmentation models
│   ├── association_rules/     # Medical association mining
│   ├── cnn/                   # Imaging diagnostics models
│   ├── rnn_lstm/              # Sequence modeling for vitals
│   └── nlp/                   # NLP models (BioBERT, sentiment, translation)
├── chatbot/                   # Healthcare chatbot implementation
├── api/                       # FastAPI backend
├── dashboard/                 # Streamlit dashboard
├── utils/                     # Utility functions
├── notebooks/                 # Exploratory data analysis
├── tests/                     # Unit and integration tests
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Datasets

- MIMIC-III (clinical records, vitals, diagnoses)
- PhysioNet (time-series vital signs)
- NIH Chest X-ray Dataset (image dataset for CNN)
- Patient feedback dataset

## Evaluation Metrics

- **Classification**: Accuracy, F1-score, ROC-AUC
- **Regression**: RMSE, MAE, R²
- **Clustering**: Silhouette, Calinski-Harabasz, clinical interpretability
- **Associations**: Support, Confidence, Lift
- **Imaging**: Accuracy, Precision, Recall, AUC
- **RNN/LSTM**: Forecast RMSE, early warning detection rate
- **NLP**: BLEU, COMET, F1 on NER tasks
- **Sentiment**: Precision/Recall, MCC
- **Chatbot**: Relevance score, Faithfulness, Response latency

## Technologies

- Python, PyTorch, TensorFlow
- scikit-learn, spaCy, NLTK, Hugging Face
- FastAPI, Streamlit
- Docker, MLflow