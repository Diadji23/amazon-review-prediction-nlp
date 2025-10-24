from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Tuple
import numpy as np
import joblib


class FeatureExtractor:
    """Class for feature extraction"""
    
    def __init__(self, method: str = 'tfidf', **kwargs):
        """
        Initialize the  features extractor.
        
        Args:
            method: 'tfidf' ou 'count'
            **kwargs:  vectorizern params
         """
        self.method = method
        self.vectorizer = None
        self.is_fitted = False
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),  # unigrams + bigrams
                min_df=5,
                max_df=0.8,
                **kwargs
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8,
                **kwargs
            )
        else:
            raise ValueError(f"Method unkown: {method}")
    
    def fit_transform(self, texts: List[str]):
        """
        Fit  vectorizer and transform  texte.
        
        Args:
            texts: List of texts
            
        Returns:
            Sparse matrix of features
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        print(f" extracted Feature: {features.shape}")
        return features
    
    def transform(self, texts: List[str]):
        """
        Transform texts.
        
        Args:
            texts: Liste of  textes
            
        Returns:
            Sparse matrix de features
        """
        if not self.is_fitted:
            raise ValueError("Le vectorizer must be fitted before")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Retourn feeatures names"""
        if not self.is_fitted:
            raise ValueError("Le vectorizer must be fitted before")
        
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath: str) -> None:
        """Save the vectorizer."""
        if not self.is_fitted:
            raise ValueError("fit vectorzier before ")
        
        joblib.dump(self.vectorizer, filepath)
        print(f" Vectorizer saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, method: str = 'tfidf') -> 'FeatureExtractor':
        """load a  vectorizer ."""
        instance = cls(method=method)
        instance.vectorizer = joblib.load(filepath)
        instance.is_fitted = True
        print(f" Vectorizer loaded: {filepath}")
        return instance