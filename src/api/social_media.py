# social_media.py

from fastapi import APIRouter
from pydantic import BaseModel
from ml_models.social_media_analysis import analyze_sentiment, detect_fake_news

# --- Create Router for Social Media Analysis ---
social_media_router = APIRouter()

# Input schema for social media data
class SocialMediaInput(BaseModel):
    text: str
    platform: str

# Endpoint to analyze sentiment of a given post or tweet
@social_media_router.post("/sentiment")
def analyze_sentiment_data(input_data: SocialMediaInput):
    sentiment = analyze_sentiment(input_data.text)
    return {"sentiment": sentiment}

# Endpoint to detect fake news in a post or article
@social_media_router.post("/fake_news")
def analyze_fake_news(input_data: SocialMediaInput):
    is_fake = detect_fake_news(input_data.text)
    return {"is_fake": is_fake}
