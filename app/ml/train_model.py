# app/ml/train_model.py

"""
This script trains a machine learning model to classify expense descriptions into categories.
It uses a TF-IDF vectorizer + Logistic Regression pipeline and saves the trained model to disk.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
DATA_PATH = "data/expenses.csv"
df = pd.read_csv(DATA_PATH)

# Split data
X = df["description"]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (Vectorizer + Classifier)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate model on test data
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model to disk
os.makedirs("app/models", exist_ok=True)
joblib.dump(pipeline, "app/models/expense_classifier.pkl")
print("Model saved to app/models/expense_classifier.pkl")