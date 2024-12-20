# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
nltk.download('vader_lexicon', quiet=True)  # Suppress NLTK download output
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import speech_recognition as sr
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class SentimentPipeline:
    def __init__(self, model_path=None, vectorizer_path=None, use_vader=True, use_ai_model=False):
        """
        Initialize the sentiment pipeline.
        - If `use_vader` is True, VADER will be used for sentiment analysis.
        - If `use_vader` is False and `use_ai_model` is True, an AI model will be used for sentiment analysis.
        """
        self.use_vader = use_vader
        self.use_ai_model = use_ai_model

        if use_ai_model:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.ai_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        elif not use_vader:
            if not model_path or not vectorizer_path:
                raise ValueError("Model and Vectorizer paths must be provided if not using VADER.")
            self.model = self.load_pickle(model_path)
            self.vectorizer = self.load_pickle(vectorizer_path)
        else:
            self.vader = SentimentIntensityAnalyzer()

    @staticmethod
    def load_pickle(file_path):
        """Load a pickle file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def clean_text(text):
        """Preprocess text by cleaning and normalizing."""
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict_sentiment_vader(self, text):
        """Predict sentiment using VADER."""
        scores = self.vader.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def predict_sentiment_ai_model(self, text):
        """Predict sentiment using an AI model."""
        results = self.ai_pipeline(text)
        return results[0]['label']

    def predict_sentiment_model(self, text):
        """Predict sentiment using a custom-trained model."""
        cleaned_text = self.clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        return "Positive" if prediction == 1 else "Negative"

    def process_input(self, input_data):
        """Process and predict sentiment for input data."""
        if isinstance(input_data, str):  # Single text input
            if self.use_ai_model:
                return self.predict_sentiment_ai_model(input_data)
            elif self.use_vader:
                return self.predict_sentiment_vader(input_data)
            else:
                return self.predict_sentiment_model(input_data)
        else:
            raise ValueError("Unsupported input type. Provide a string.")

class SpeechToTextSentiment:
    def __init__(self, sentiment_pipeline):
        """
        Initialize the speech-to-text sentiment analysis pipeline.
        :param sentiment_pipeline: An instance of the SentimentPipeline class.
        """
        self.recognizer = sr.Recognizer()
        self.sentiment_pipeline = sentiment_pipeline

    def speech_to_text_from_file(self, audio_file_path):
        """Convert speech input from a .wav file to text."""
        try:
            with sr.AudioFile(audio_file_path) as source:
                print("Processing audio file...")
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                print(f"Transcribed text: {text}")
                return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

    def analyze_file_sentiment(self, audio_file_path):
        """Analyze sentiment from a .wav file."""
        text = self.speech_to_text_from_file(audio_file_path)
        if text:
            sentiment = self.sentiment_pipeline.process_input(text)
            print(f"Sentiment: {sentiment}")
            return sentiment
        else:
            print("No valid audio input detected.")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the sentiment pipeline using an AI model
    sentiment_pipeline = SentimentPipeline(use_vader=False, use_ai_model=True)

    # Initialize the speech-to-text sentiment analysis system
    stt_sentiment = SpeechToTextSentiment(sentiment_pipeline)

    # Path to the .wav file
    audio_file_path = "download.wav"

    # Check if the file exists
    if not os.path.isfile(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
    else:
        # Analyze sentiment from the .wav file
        print("Starting sentiment analysis...")
        sentiment = stt_sentiment.analyze_file_sentiment(audio_file_path)
        if sentiment:
            print(f"Final Sentiment: {sentiment}")
        else:
            print("No sentiment analysis could be performed.")
