# main.py

from fastapi import FastAPI
from geospatial import geospatial_router
from social_media import social_media_router
from cctv import cctv_router
from fake_document import fake_document_router

# --- Create the FastAPI app ---
app = FastAPI()

# --- Include Routers for Each AI Model ---
app.include_router(geospatial_router, prefix="/geospatial", tags=["Geospatial"])
app.include_router(social_media_router, prefix="/social_media", tags=["Social Media"])
app.include_router(cctv_router, prefix="/cctv", tags=["CCTV"])
app.include_router(fake_document_router, prefix="/fake_document", tags=["Fake Document"])

# Example route for testing API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Immigration Monitoring System API"}
