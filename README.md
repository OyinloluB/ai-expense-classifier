# AI-Powered Expense Classifier

An intelligent API that classifies natural-language expense descriptions into categories like `Transportation`, `Food & Drink`, `Entertainment`, etc powered by machine learning and FastAPI.

It also supports:
- Batch classification
- OCR classification from images/receipts
- Confidence scoring for each prediction

---

## Features

- **Text Classification** — Predicts expense categories from written descriptions.
- **OCR Support** — Extracts and classifies text from images (e.g. receipts).
- **ML-Powered** — Trained using scikit-learn with `TfidfVectorizer` and `LogisticRegression`.
- **Batch Input** — Classify multiple expenses in a single call.
- **FastAPI Backend** — Modern async API with auto-generated Swagger docs.
  
---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-expense-classifier.git
cd ai-expense-classifier
```

### 2. Create and activate virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install fastapi uvicorn scikit-learn pandas joblib pillow pytesseract python-multipart
```

### 4. Model training

```
python app/ml/train_model.py
```

This script trains a logistic regression classifier on the mock dataset and saves it to `app/models/expense_classifier.pkl`

### 5. Run the API

```
uvicorn app.main:app --reload
```

Visit:

http://localhost:8000

http://localhost:8000/docs (Swagger UI)


### Example usage

```
POST /classify
{
  "description": "Flight to Toronto"
}
```

Response

```
{
  "description": "Flight to Toronto",
  "predicted_category": "Travel",
  "confidence": 0.84
}
```
