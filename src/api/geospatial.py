# geospatial.py

from fastapi import APIRouter
from pydantic import BaseModel
from ml_models.geospatial_analysis import run_geospatial_analysis  # Importing the function from ML models

# --- Create Router for Geospatial AI ---
geospatial_router = APIRouter()

# Input schema for Geospatial model
class GeospatialInput(BaseModel):
    latitude: float
    longitude: float
    date: str  # For time-based analysis

# Example of an endpoint for geospatial analysis
@geospatial_router.post("/analyze")
def analyze_geospatial_data(input_data: GeospatialInput):
    analysis_result = run_geospatial_analysis(input_data.latitude, input_data.longitude, input_data.date)
    return {"result": analysis_result}
