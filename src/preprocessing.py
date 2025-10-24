import re 
import string 
from typing import List  
import numpy as np 


#cleaning 


def clean_text(text): 

    """ 
    clean a text : lowercase , remove ponctuation etc 

    args:  
            text :  raw text to  clean 
    returns : 
             cleaned text 
    """

    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_reviews(reviews: List[str]) -> List[str]:
    """Apply cleaning on a  list of reviews"""
    return [clean_text(review) for review in reviews]


def remove_stopwords(text: str, stopwords: set = None) -> str:
    """
    delete  stopwords of a  text.
    
    Args:
        text: cleaned Text
        stopwords: Set de stopwords 
        
    Returns:
        Texte without  stopwords
    """
    if stopwords is None:
        # Stopwords  in english
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'
        }
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)


def get_text_stats(text: str) -> dict:
    """
     statistics 
    
    Args:
        text: Texte 
        
    Returns:
        Dictionnary of  stats (nb_words, nb_chars, avg_word_length)
    """
    words = text.split()
    return {
        'nb_words': len(words),
        'nb_chars': len(text),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0
    }