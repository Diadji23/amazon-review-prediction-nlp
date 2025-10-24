from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn

from src.preprocessing import clean_text

app = FastAPI(
    title="Amazon Sentiment Analysis API",
    description="Predicting customer review sentiment",
    version="1.0.0"
)

# Load model 
try:
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model, vectorizer = None, None



class Review(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is absolutely amazing! Best purchase ever."
            }
        }


class BatchReviews(BaseModel):
    reviews: List[str]


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# Endpoints
@app.get("/")
def root():
    """Root endpoint - API home page"""
    return {
        "message": "Amazon Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict sentiment for a single review",
            "/predict/batch": "POST - Predict sentiment for multiple reviews",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive documentation"
        }
    }


@app.get("/health")
def health_check():
    """Check if the API and model are operational"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(review: Review):
    """
    Predict the sentiment of a customer review.

    - **text**: The review text to analyze.

    Returns:
    - **sentiment**: "positive" or "negative"
    - **confidence**: Confidence score between 0 and 1
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocessing
        cleaned_text = clean_text(review.text)

        # Vectorization
        text_vector = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0]

        # Format result
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(proba[prediction])

        return PredictionResponse(
            text=review.text,
            sentiment=sentiment,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchReviews):
    """
    Predict the sentiment of multiple reviews in a single request.

    - **reviews**: List of review texts to analyze.

    Returns a list of predictions.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(batch.reviews) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 reviews per batch")

    try:
        predictions = []

        for text in batch.reviews:
            cleaned = clean_text(text)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]

            sentiment = "positive" if pred == 1 else "negative"
            confidence = float(proba[pred])

            predictions.append(
                PredictionResponse(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence
                )
            )

        return BatchPredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/stats")
def get_model_stats():
    """Return basic model statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "Logistic Regression",
        "vectorizer": "TF-IDF (max_features=10000, ngram_range=(1,2))",
        "accuracy": 0.9471,
        "f1_score": 0.9691,
        "precision": 0.9572,
        "recall": 0.9812,
        "training_samples": 420651,
        "test_samples": 105163,
        "features": 10000,
        "class_distribution": {
            "negative": 16407,
            "positive": 88756
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
