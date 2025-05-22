# cctv.py

from fastapi import APIRouter
from pydantic import BaseModel
from ml_models.cctv_analysis import detect_intrusion

# --- Create Router for CCTV Analysis ---
cctv_router = APIRouter()

# Input schema for CCTV analysis
class CCTVInput(BaseModel):
    video_path: str  # Path to the video feed or camera feed
    timestamp: str   # Time of the event

# Endpoint to detect intrusion or unauthorized movement in CCTV footage
@cctv_router.post("/intrusion")
def analyze_intrusion(input_data: CCTVInput):
    intrusion_result = detect_intrusion(input_data.video_path, input_data.timestamp)
    return {"intrusion_detected": intrusion_result}
