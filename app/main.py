# app/main.py

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Expense Classifier",
    description="An ML-powered API for classifying expense descriptions and receipt images.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "AI Expense Classifier is up and running!"}

app.include_router(router)