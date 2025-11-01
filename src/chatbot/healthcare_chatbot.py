"""
Healthcare Chatbot with RAG Pipeline

This module implements a healthcare chatbot that uses Retrieval-Augmented Generation (RAG)
to provide accurate and relevant responses to patient queries about symptoms, medications,
and general health information.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class HealthcareChatbot:
    """
    Healthcare Chatbot with RAG (Retrieval-Augmented Generation) pipeline
    """
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the healthcare chatbot
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize FAQ responses
        self.faq_responses = self._initialize_faq_responses()
        
        # Fit the vectorizer on knowledge base
        self._fit_vectorizer()
        
    def _load_knowledge_base(self, path: str) -> List[Dict]:
        """
        Load the healthcare knowledge base
        
        Args:
            path: Path to knowledge base file
            
        Returns:
            List of knowledge base entries
        """
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default knowledge base
            return self._get_default_knowledge_base()
    
    def _get_default_knowledge_base(self) -> List[Dict]:
        """
        Get default healthcare knowledge base
        
        Returns:
            Default knowledge base entries
        """
        return [
            {
                "category": "symptoms",
                "keywords": ["fever", "headache", "cough", "sore throat", "nausea", "vomiting", "diarrhea"],
                "question": "What should I do if I have fever?",
                "answer": "If you have a fever above 100.4°F (38°C), rest, drink plenty of fluids, and consider taking fever-reducing medication. Contact a healthcare provider if the fever persists for more than 3 days or is accompanied by severe symptoms.",
                "urgency": "moderate"
            },
            {
                "category": "symptoms",
                "keywords": ["chest pain", "shortness of breath", "difficulty breathing"],
                "question": "What should I do if I have chest pain?",
                "answer": "Chest pain can be serious. If you experience chest pain, especially if it's severe, persistent, or accompanied by shortness of breath, nausea, or sweating, seek immediate medical attention or call emergency services.",
                "urgency": "high"
            },
            {
                "category": "medications",
                "keywords": ["medication", "drug", "prescription", "dosage"],
                "question": "How should I take my medications?",
                "answer": "Always follow your doctor's instructions and read medication labels carefully. Take medications at the same time each day, with or without food as directed. Never stop taking prescribed medications without consulting your healthcare provider.",
                "urgency": "low"
            },
            {
                "category": "appointments",
                "keywords": ["appointment", "schedule", "visit", "doctor"],
                "question": "How can I schedule an appointment?",
                "answer": "You can schedule an appointment by calling our main number, using our online patient portal, or speaking with our reception staff. Please have your insurance information and preferred dates ready.",
                "urgency": "low"
            },
            {
                "category": "emergency",
                "keywords": ["emergency", "urgent", "911", "ambulance"],
                "question": "When should I call 911?",
                "answer": "Call 911 for life-threatening emergencies such as severe chest pain, difficulty breathing, loss of consciousness, severe bleeding, or signs of stroke. For non-emergency concerns, contact your healthcare provider or visit urgent care.",
                "urgency": "high"
            },
            {
                "category": "prevention",
                "keywords": ["prevention", "vaccine", "immunization", "healthy lifestyle"],
                "question": "How can I prevent illness?",
                "answer": "Maintain a healthy lifestyle by eating a balanced diet, exercising regularly, getting adequate sleep, managing stress, staying hydrated, and keeping up with recommended vaccinations and health screenings.",
                "urgency": "low"
            }
        ]
    
    def _initialize_faq_responses(self) -> Dict[str, str]:
        """
        Initialize frequently asked questions and responses
        
        Returns:
            Dictionary of FAQ responses
        """
        return {
            "hours": "Our clinic is open Monday through Friday from 8:00 AM to 6:00 PM, and Saturday from 9:00 AM to 2:00 PM. We are closed on Sundays and major holidays.",
            "location": "We are located at 123 Healthcare Drive, Medical City, MC 12345. We have ample parking and are easily accessible by public transportation.",
            "insurance": "We accept most major insurance plans including Medicare, Medicaid, and private insurance. Please contact our billing department to verify your specific coverage.",
            "prescription": "You can request prescription refills through our patient portal, by calling our pharmacy line, or by contacting your doctor's office directly. Please allow 24-48 hours for processing.",
            "test results": "Test results are typically available within 3-5 business days. You can access your results through our patient portal or by calling our office. Your doctor will contact you if immediate follow-up is needed.",
            "billing": "You can pay your bill online through our patient portal, by mail, or by calling our billing department. We also offer payment plans for qualifying patients."
        }
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on the knowledge base"""
        texts = []
        for entry in self.knowledge_base:
            texts.append(entry['question'] + ' ' + ' '.join(entry['keywords']))
        
        self.knowledge_vectors = self.vectorizer.fit_transform(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess input text
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant knowledge base entries for a query
        
        Args:
            query: User query
            top_k: Number of top relevant entries to return
            
        Returns:
            List of relevant knowledge base entries
        """
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        
        # Get top-k most similar entries
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_entries = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                entry = self.knowledge_base[idx].copy()
                entry['similarity'] = similarities[idx]
                relevant_entries.append(entry)
        
        return relevant_entries
    
    def _generate_response(self, query: str, relevant_entries: List[Dict]) -> str:
        """
        Generate response based on retrieved knowledge
        
        Args:
            query: User query
            relevant_entries: Relevant knowledge base entries
            
        Returns:
            Generated response
        """
        if not relevant_entries:
            return "I'm sorry, I don't have specific information about that. Please consult with a healthcare provider for medical advice."
        
        # Get the most relevant entry
        best_entry = relevant_entries[0]
        
        # Check for urgency
        if best_entry.get('urgency') == 'high':
            response = f"⚠️ IMPORTANT: {best_entry['answer']}\n\n"
            response += "This appears to be a serious medical concern. Please seek immediate medical attention if needed."
        else:
            response = f"{best_entry['answer']}\n\n"
        
        # Add disclaimer
        response += "\n\n⚠️ Disclaimer: This information is for general educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for specific medical concerns."
        
        return response
    
    def _check_faq(self, query: str) -> Optional[str]:
        """
        Check if query matches any FAQ
        
        Args:
            query: User query
            
        Returns:
            FAQ response if found, None otherwise
        """
        query_lower = query.lower()
        
        for faq_key, faq_response in self.faq_responses.items():
            if faq_key in query_lower:
                return faq_response
        
        return None
    
    def chat(self, user_input: str) -> str:
        """
        Main chat function
        
        Args:
            user_input: User's input message
            
        Returns:
            Chatbot response
        """
        # Check FAQ first
        faq_response = self._check_faq(user_input)
        if faq_response:
            return faq_response
        
        # Retrieve relevant knowledge
        relevant_entries = self._retrieve_relevant_knowledge(user_input)
        
        # Generate response
        response = self._generate_response(user_input, relevant_entries)
        
        return response
    
    def add_knowledge_entry(self, category: str, keywords: List[str], 
                          question: str, answer: str, urgency: str = "low"):
        """
        Add a new knowledge base entry
        
        Args:
            category: Category of the knowledge
            keywords: Keywords for matching
            question: Question this entry answers
            answer: Answer to provide
            urgency: Urgency level (low, moderate, high)
        """
        new_entry = {
            "category": category,
            "keywords": keywords,
            "question": question,
            "answer": answer,
            "urgency": urgency
        }
        
        self.knowledge_base.append(new_entry)
        self._fit_vectorizer()  # Refit vectorizer with new data
    
    def get_statistics(self) -> Dict:
        """
        Get chatbot statistics
        
        Returns:
            Dictionary with chatbot statistics
        """
        return {
            "total_knowledge_entries": len(self.knowledge_base),
            "categories": list(set(entry['category'] for entry in self.knowledge_base)),
            "high_urgency_entries": len([e for e in self.knowledge_base if e.get('urgency') == 'high']),
            "faq_count": len(self.faq_responses)
        }


def main():
    """
    Main function to demonstrate the chatbot
    """
    # Initialize chatbot
    chatbot = HealthcareChatbot()
    
    print("🏥 Healthcare Chatbot")
    print("=" * 50)
    print("Ask me about symptoms, medications, appointments, or general health questions.")
    print("Type 'quit' to exit.\n")
    
    # Get chatbot statistics
    stats = chatbot.get_statistics()
    print(f"Knowledge Base: {stats['total_knowledge_entries']} entries")
    print(f"Categories: {', '.join(stats['categories'])}")
    print(f"FAQ Responses: {stats['faq_count']}\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Thank you for using our healthcare chatbot. Take care!")
            break
        
        if not user_input:
            continue
        
        response = chatbot.chat(user_input)
        print(f"Chatbot: {response}\n")


if __name__ == "__main__":
    main()
