
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_reviews
from src.features import FeatureExtractor
from src.models import SentimentClassifier
import pandas as pd
import numpy as np


def train_pipeline(data_path: str, save_dir: str = 'models'):
    """
    training pipeline     
    Args:
        data_path: path to csv data files 
        save_dir: where to save the trained model
    """
    print("="*60)
    print("PIPELINE D'ENTRAÎNEMENT - SENTIMENT ANALYSIS")
    print("="*60)
    
    # data loading 
    print("\n[1/5] Loading ...")
    df = pd.read_csv(data_path)
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()  # 0=negative, 1=positive
    
    
    # Preprocessing
    print("\n[2/5]text Preprocessing...")
    texts_clean = preprocess_reviews(texts)
    print(f" Texts cleaned ")
    
    # Split train/test
    print("\n[3/5] Split train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f" Train: {len(X_train_text)} | Test: {len(X_test_text)}")
    
    #  Feature extraction
    print("\n[4/5] Extraction de features (TF-IDF)...")
    feature_extractor = FeatureExtractor(method='tfidf')
    X_train = feature_extractor.fit_transform(X_train_text)
    X_test = feature_extractor.transform(X_test_text)
    
    #  mdoel training 
    print("\n[5/5] Model training...")
    classifier = SentimentClassifier(model_type='logistic')
    classifier.train(X_train, y_train)
    
    # evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    metrics = classifier.evaluate(X_test, y_test)
    
    #  Save
    print("\n" + "="*60)
    print("SAVE")
    print("="*60)
    classifier.save(f'{save_dir}/sentiment_model.pkl')
    feature_extractor.save(f'{save_dir}/tfidf_vectorizer.pkl')
    
    print("\n Pipeline Finished")
    return classifier, feature_extractor, metrics


if __name__ == "__main__":
    train_pipeline(data_path="data/reviews.csv")