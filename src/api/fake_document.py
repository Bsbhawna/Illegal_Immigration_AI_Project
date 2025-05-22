# fake_document.py

from fastapi import APIRouter
from pydantic import BaseModel
from ml_models.fake_document_detection import detect_fake_document

# --- Create Router for Fake Document Detection ---
fake_document_router = APIRouter()

# Input schema for fake document detection
class DocumentInput(BaseModel):
    document_path: str  # Path to the document (could be image or PDF)
    document_type: str  # Type of document (e.g., Passport, ID card)

# Endpoint to detect fake document
@fake_document_router.post("/detect")
def detect_fake(input_data: DocumentInput):
    result = detect_fake_document(input_data.document_path, input_data.document_type)
    return {"is_fake": result}
