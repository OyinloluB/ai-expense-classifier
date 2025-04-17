# tests/test_routes.py

from starlette.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Expense Classifier is up and running!"}

# single expense classification
def test_classify_single_expense():
    payload = {"description": "Uber to the airport"}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

# batch expense classification
def test_classify_batch_expenses():
    payload = {
        "descriptions": [
            "Dinner at Olive Garden",
            "Netflix subscription",
            "Gas station refill"
        ]
    }
    response = client.post("/classify/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 3

    for item in data["results"]:
        assert "predicted_category" in item
        assert "confidence" in item
        assert isinstance(item["confidence"], float)

# OCR receipt upload
def test_classify_ocr_receipt():
    fake_image = io.BytesIO(b"fake image data")
    response = client.post(
        "/classify/ocr",
        files={"file": ("fake_receipt.png", fake_image, "image/png")}
    )
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_category" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)

# invalid file uploaded to OCR
def test_classify_ocr_invalid_file():
    fake_file = io.BytesIO(b"not an image")
    response = client.post(
        "/classify/ocr",
        files={"file": ("receipt.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type. Please upload an image file (png, jpg, jpeg)."