"""
Medical Translator

This module implements a machine translator for multilingual medical communication
using MarianMT for English ↔ regional language translation.
"""

import os
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

class MedicalTranslator:
    """
    Medical translator for multilingual healthcare communication
    """
    
    def __init__(self):
        """Initialize the medical translator"""
        self.models = {}
        self.tokenizers = {}
        self.medical_glossary = self._load_medical_glossary()
        
        # Supported language pairs
        self.supported_languages = {
            "es": "Spanish",
            "fr": "French", 
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        
        # MarianMT model mappings
        self.model_mappings = {
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr", 
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "en-it": "Helsinki-NLP/opus-mt-en-it",
            "it-en": "Helsinki-NLP/opus-mt-it-en",
            "en-pt": "Helsinki-NLP/opus-mt-en-pt",
            "pt-en": "Helsinki-NLP/opus-mt-pt-en",
            "en-ru": "Helsinki-NLP/opus-mt-en-ru",
            "ru-en": "Helsinki-NLP/opus-mt-ru-en",
            "en-zh": "Helsinki-NLP/opus-mt-en-zh",
            "zh-en": "Helsinki-NLP/opus-mt-zh-en",
            "en-ja": "Helsinki-NLP/opus-mt-en-ja",
            "ja-en": "Helsinki-NLP/opus-mt-ja-en",
            "en-ko": "Helsinki-NLP/opus-mt-en-ko",
            "ko-en": "Helsinki-NLP/opus-mt-ko-en",
            "en-ar": "Helsinki-NLP/opus-mt-en-ar",
            "ar-en": "Helsinki-NLP/opus-mt-ar-en",
            "en-hi": "Helsinki-NLP/opus-mt-en-hi",
            "hi-en": "Helsinki-NLP/opus-mt-hi-en"
        }
    
    def _load_medical_glossary(self) -> Dict[str, Dict[str, str]]:
        """
        Load medical terminology glossary
        
        Returns:
            Dictionary of medical terms in different languages
        """
        return {
            "en": {
                "pain": "pain",
                "headache": "headache", 
                "fever": "fever",
                "cough": "cough",
                "nausea": "nausea",
                "vomiting": "vomiting",
                "diarrhea": "diarrhea",
                "chest pain": "chest pain",
                "shortness of breath": "shortness of breath",
                "dizziness": "dizziness",
                "fatigue": "fatigue",
                "medicine": "medicine",
                "prescription": "prescription",
                "doctor": "doctor",
                "nurse": "nurse",
                "hospital": "hospital",
                "emergency": "emergency",
                "appointment": "appointment",
                "blood pressure": "blood pressure",
                "heart rate": "heart rate",
                "temperature": "temperature"
            },
            "es": {
                "pain": "dolor",
                "headache": "dolor de cabeza",
                "fever": "fiebre", 
                "cough": "tos",
                "nausea": "náusea",
                "vomiting": "vómito",
                "diarrhea": "diarrea",
                "chest pain": "dolor de pecho",
                "shortness of breath": "dificultad para respirar",
                "dizziness": "mareo",
                "fatigue": "fatiga",
                "medicine": "medicina",
                "prescription": "receta",
                "doctor": "doctor",
                "nurse": "enfermera",
                "hospital": "hospital",
                "emergency": "emergencia",
                "appointment": "cita",
                "blood pressure": "presión arterial",
                "heart rate": "frecuencia cardíaca",
                "temperature": "temperatura"
            },
            "fr": {
                "pain": "douleur",
                "headache": "mal de tête",
                "fever": "fièvre",
                "cough": "toux", 
                "nausea": "nausée",
                "vomiting": "vomissement",
                "diarrhea": "diarrhée",
                "chest pain": "douleur thoracique",
                "shortness of breath": "essoufflement",
                "dizziness": "étourdissement",
                "fatigue": "fatigue",
                "medicine": "médicament",
                "prescription": "ordonnance",
                "doctor": "médecin",
                "nurse": "infirmière",
                "hospital": "hôpital",
                "emergency": "urgence",
                "appointment": "rendez-vous",
                "blood pressure": "tension artérielle",
                "heart rate": "fréquence cardiaque",
                "temperature": "température"
            }
        }
    
    def _load_model(self, source_lang: str, target_lang: str) -> bool:
        """
        Load translation model for given language pair
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            return False
            
        model_key = f"{source_lang}-{target_lang}"
        
        if model_key not in self.model_mappings:
            return False
        
        try:
            model_name = self.model_mappings[model_key]
            
            # Load model and tokenizer
            self.tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
            self.models[model_key] = MarianMTModel.from_pretrained(model_name)
            
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def _preprocess_medical_text(self, text: str) -> str:
        """
        Preprocess medical text for better translation
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase for medical terms lookup
        text_lower = text.lower()
        
        # Replace common medical abbreviations
        abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate", 
            "temp": "temperature",
            "rx": "prescription",
            "pt": "patient",
            "dr": "doctor"
        }
        
        for abbrev, full_form in abbreviations.items():
            text = text.replace(abbrev, full_form)
            text_lower = text_lower.replace(abbrev, full_form)
        
        return text
    
    def _postprocess_translation(self, text: str, target_lang: str) -> str:
        """
        Postprocess translation for medical context
        
        Args:
            text: Translated text
            target_lang: Target language code
            
        Returns:
            Postprocessed text
        """
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "es") -> str:
        """
        Translate medical text between languages
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
        
        # Check if languages are supported
        if source_lang not in self.supported_languages or target_lang not in self.supported_languages:
            return f"Language pair {source_lang}-{target_lang} not supported"
        
        # If same language, return original text
        if source_lang == target_lang:
            return text
        
        model_key = f"{source_lang}-{target_lang}"
        
        # Load model if not already loaded
        if model_key not in self.models:
            if not self._load_model(source_lang, target_lang):
                return f"Translation model for {source_lang}-{target_lang} not available"
        
        try:
            # Preprocess text
            processed_text = self._preprocess_medical_text(text)
            
            # Tokenize
            inputs = self.tokenizers[model_key](processed_text, return_tensors="pt", padding=True, truncation=True)
            
            # Translate
            with torch.no_grad():
                translated = self.models[model_key].generate(**inputs)
            
            # Decode
            translated_text = self.tokenizers[model_key].decode(translated[0], skip_special_tokens=True)
            
            # Postprocess
            translated_text = self._postprocess_translation(translated_text, target_lang)
            
            return translated_text
            
        except Exception as e:
            return f"Translation error: {str(e)}"
    
    def translate_medical_term(self, term: str, source_lang: str = "en", target_lang: str = "es") -> str:
        """
        Translate medical terms using glossary
        
        Args:
            term: Medical term to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated medical term
        """
        term_lower = term.lower()
        
        # Check glossary first
        if source_lang in self.medical_glossary and target_lang in self.medical_glossary:
            if term_lower in self.medical_glossary[source_lang]:
                source_terms = self.medical_glossary[source_lang]
                target_terms = self.medical_glossary[target_lang]
                
                # Find matching term
                for key, value in source_terms.items():
                    if value.lower() == term_lower:
                        if key in target_terms:
                            return target_terms[key]
        
        # Fallback to general translation
        return self.translate(term, source_lang, target_lang)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages
        
        Returns:
            Dictionary of language codes and names
        """
        return self.supported_languages.copy()
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available translation models
        
        Returns:
            List of available model keys
        """
        return list(self.model_mappings.keys())
    
    def add_medical_term(self, term: str, translation: str, source_lang: str, target_lang: str):
        """
        Add medical term to glossary
        
        Args:
            term: Medical term
            translation: Translation of the term
            source_lang: Source language code
            target_lang: Target language code
        """
        if source_lang not in self.medical_glossary:
            self.medical_glossary[source_lang] = {}
        if target_lang not in self.medical_glossary:
            self.medical_glossary[target_lang] = {}
        
        self.medical_glossary[source_lang][term.lower()] = term
        self.medical_glossary[target_lang][term.lower()] = translation


def main():
    """
    Main function to demonstrate the translator
    """
    translator = MedicalTranslator()
    
    print("🌍 Medical Translator")
    print("=" * 50)
    print("Available languages:", ", ".join(translator.get_supported_languages().values()))
    print()
    
    # Example translations
    examples = [
        ("I have a headache and fever", "en", "es"),
        ("Tengo dolor de pecho", "es", "en"),
        ("I need to see a doctor", "en", "fr"),
        ("J'ai mal à la tête", "fr", "en")
    ]
    
    for text, source, target in examples:
        translated = translator.translate(text, source, target)
        print(f"{text} ({source}) → {translated} ({target})")
    
    print("\nMedical terms:")
    medical_terms = ["headache", "fever", "chest pain", "blood pressure"]
    
    for term in medical_terms:
        spanish = translator.translate_medical_term(term, "en", "es")
        french = translator.translate_medical_term(term, "en", "fr")
        print(f"{term} → Spanish: {spanish}, French: {french}")


if __name__ == "__main__":
    main()
