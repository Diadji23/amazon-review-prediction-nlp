# Amazon Review Sentiment Analysis API

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/accuracy-94.7%25-brightgreen.svg)](.)

Production-ready sentiment classification system for Amazon customer reviews with **REST API deployment**.

>  **Project 2025** — Data Science & NLP  
> 

---

##  Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **94.71%** |
| **F1-Score** | **0.9691** |
| **Precision** | 0.9572 |
| **Recall** | 0.9812 |

**Dataset:** 525,814 Amazon reviews (420k train / 105k test)  
**Features:** TF-IDF (10,000 features, n-grams 1-2)  
**Model:** Logistic Regression with L2 regularization

---

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Diadji23/amazon-review-prediction-nlp.git
cd amazon-review-prediction-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Run complete training pipeline
python src/train.py

# Models will be saved to models/
```

### Launch API

```bash
# Start server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# API available at:
# http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Quick Test

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This product is absolutely amazing!"}
)

print(response.json())
# Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.96}
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Great product, highly recommend!"}'
```

---

##  Project Architecture

```
amazon-review-prediction-nlp/
│
├── api.py                      # FastAPI REST API
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
│
├── src/                        # Source code
│   ├── preprocessing.py        # Text cleaning (lowercase, punctuation, etc.)
│   ├── features.py             # TF-IDF vectorization
│   ├── models.py               # Logistic Regression classifier
│   ├── train.py                # Training pipeline
│   └── utils.py                # Utility functions
│
├── models/                     # Saved models (.pkl)
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/                  # Jupyter experiments
│   └── (exploratory notebooks)
│
├── tests/                      # Unit tests
│   ├── test_api.py
│   └── test_preprocessing.py
│
└── results/                    # Results & visualizations
    └── screenshots/
```

---

##  Technical Approach

### 1. Preprocessing

- **Cleaning:** Lowercase, URL/email/punctuation removal
- **Normalization:** Whitespace, special characters
- **Preservation:** Negations (crucial for sentiment)

```python
Input:  "This is AMAZING!!! http://example.com"
Output: "this is amazing"
```

### 2. Feature Engineering

**TF-IDF Vectorization:**
- Max features: 10,000
- N-grams: (1, 2) — unigrams + bigrams
- Min/max doc frequency: 5 / 0.8
- Output: Sparse matrix (525k × 10k)

### 3. Model

**Logistic Regression:**
- Solver: lbfgs
- Max iterations: 1000
- L2 regularization
- Training time: ~2 minutes
- Inference time: ~20ms per prediction

**Why Logistic Regression?**
-  Fast to train and predict
-  Interpretable (coefficients = word importance)
-  Excellent performance on text classification
-  Easy to deploy in production

---

##  API Endpoints

### `POST /predict`
Predict sentiment for a single review.

**Request:**
```json
{"text": "Amazing product!"}
```

**Response:**
```json
{
  "text": "Amazing product!",
  "sentiment": "positive",
  "confidence": 0.96
}
```

### `POST /predict/batch`
Predict multiple reviews at once (max 100).

**Request:**
```json
{
  "reviews": ["Great!", "Terrible...", "It's okay"]
}
```

**Response:**
```json
{
  "predictions": [
    {"text": "Great!", "sentiment": "positive", "confidence": 0.94},
    {"text": "Terrible...", "sentiment": "negative", "confidence": 0.89},
    {"text": "It's okay", "sentiment": "positive", "confidence": 0.62}
  ]
}
```

### `GET /health`
Check API status.

**Response:**
```json
{"status": "healthy", "model_loaded": true}
```

### `GET /stats`
Model statistics and metrics.

**Response:**
```json
{
  "model_type": "Logistic Regression",
  "accuracy": 0.9471,
  "f1_score": 0.9691,
  "precision": 0.9572,
  "recall": 0.9812,
  "training_samples": 420651,
  "test_samples": 105163
}
```

### `GET /docs`
Interactive Swagger documentation.

---

##  Docker Deployment

```bash
# Build image
docker build -t sentiment-api .

# Run container
docker run -d -p 8000:8000 sentiment-api

# With Docker Compose
docker-compose up -d
```

**Access API:**
- Local: http://localhost:8000
- Docs: http://localhost:8000/docs

---

##  Detailed Results

### Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative (0) | 0.88 | 0.76 | 0.82 | 16,407 |
| Positive (1) | 0.96 | 0.98 | 0.97 | 88,756 |

### Confusion Matrix

```
Predicted →      Negative   Positive
True Negative     12,511     3,896
True Positive      1,666    87,090
```

### Key Insights

 **Strengths:**
- Excellent precision on positive reviews (96%)
- Low false positive rate (3.7%)
- Fast inference time (<20ms)
- Robust to varying text lengths

 **Limitations:**
- Imbalanced dataset (84% positive)
- Lower performance on negative reviews (recall 76%)
- Struggles with sarcasm and irony
- Mixed reviews tend to be classified as positive

### Error Analysis

**False Negatives (predicted positive, actually negative):**
- "This product is great... if you want to waste your money!" → Sarcasm
- "Not bad, but overpriced for what it is" → Nuanced opinion

**False Positives (predicted negative, actually positive):**
- "This is not terrible at all!" → Double negation
- "Could be worse, actually works well" → Complex phrasing

---

##  Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov=api --cov-report=html

# Run specific test
pytest tests/test_api.py::test_predict_positive -v
```

**Current coverage:** 85% (preprocessing + models + API endpoints)

---

##  Tech Stack

**Machine Learning:**  
Python 3.9 • scikit-learn 1.3.0 • pandas • numpy • joblib

**API & Deployment:**  
FastAPI 0.100.0 • Uvicorn • Pydantic • Docker

**NLP Tools:**  
TF-IDF Vectorization • Text preprocessing • Regex

**Testing & QA:**  
pytest • pytest-cov • httpx

---

##  Future Improvements

### Short-term
- [ ] Complete unit tests (>90% coverage)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Performance benchmarking
- [ ] Error logging system

### Medium-term
- [ ] Compare LSTM/BERT architectures
- [ ] Multilingual support (French, Spanish)
- [ ] Aspect-based sentiment (price, quality, delivery)
- [ ] Monitoring with Prometheus/Grafana

### Long-term
- [ ] A/B testing framework
- [ ] Model explainability (LIME/SHAP)
- [ ] Active learning pipeline
- [ ] Real-time drift detection

---

##  Dataset

**Source:** Amazon Customer Reviews Dataset (public)

**Characteristics:**
- 525,814 customer reviews
- Various products (electronics, books, clothing, etc.)
- English language
- Binary labels: 0 (negative) / 1 (positive)
- Time period: 2010-2018

**Preprocessing:**
- HTML/markdown cleaning
- Filtering short reviews (<10 words)
- Deduplication
- Partial class balancing

---

##  Author

**Papa Diadji BOYE**  
Engineering Student — ENSIIE (Class of 2026)  
Specialization: Data Science, NLP, Computer Vision


##  License

MIT License — See [LICENSE](LICENSE) for details.

---


