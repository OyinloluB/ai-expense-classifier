# app/api/route.py

"""
defines the API routes for the AI Expense Classifier.
- /classify: Single expense classification
- /classify/batch: Batch classification
- /classify/ocr: Image-based OCR classification
"""

from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from typing import List
import pytesseract
import joblib
import io

# Schema for single expense input
class ExpenseInput(BaseModel):
    description: str

# Load the trained model once when the app starts
MODEL_PATH = "app/models/expense_classifier.pkl"
model = joblib.load(MODEL_PATH)

# Create the router
router = APIRouter()

# Classify single expense
@router.post("/classify")
def classify_expense(data: ExpenseInput):
    description = data.description
    prediction = model.predict([description])[0]
    confidence = max(model.predict_proba([description])[0])

    return {
        "description": description,
        "predicted_category": prediction,
        "confidence": round(confidence, 4)
    }

# Schema for batch classification input
class BatchInput(BaseModel):
    descriptions: List[str]

# Classify a batch of expense descriptions
@router.post("/classify/batch")
def classify_batch(data: BatchInput):
    descriptions = data.descriptions
    predictions = model.predict(descriptions)
    confidences = model.predict_proba(descriptions)

    results = []
    for i, desc in enumerate(descriptions):
        results.append({
            "description": desc,
            "predicted_category": predictions[i],
            "confidence": round(max(confidences[i]), 4)
        })
    
    return {"results": results}

# Classify OCR image/text/category
@router.post("/classify/ocr")
async def classify_ocr(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run OCR
        extracted_text = pytesseract.image_to_string(image)
        
        # Predict
        prediction = model.predict([extracted_text])[0]
        confidence = max(model.predict_proba([extracted_text])[0])
        
        return {
            "extracted_text": extracted_text.strip(),
            "predicted_category": prediction,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        return {"error": str(e)}