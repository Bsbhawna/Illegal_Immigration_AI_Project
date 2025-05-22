import json
import os

# ✅ Project Root
project_root = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project"

# ✅ Config Dictionary
config_data = {
    "cctv_alerts": ["src/cctv_analysis/dashboard_ready/cctv_alerts.json"],
    "twitter_alerts": [
        "src/social_media_analysis/twitter/dashboard_ready/twitter_entity_alerts.json",
        "src/social_media_analysis/twitter/dashboard_ready/twitter_sentiment_alerts.json",
        "src/social_media_analysis/twitter/dashboard_ready/twitter_sentiment_summary.json"
    ],
    "reddit_alerts": [
        "src/social_media_analysis/reddit/dashboard_ready/reddit_entity_alerts.json",
        "src/social_media_analysis/reddit/dashboard_ready/reddit_sentiment_alerts.json"
    ],
    "darkweb_alerts": ["data/processed/darkweb_alerts.json"],
    "integrated_alerts_output": "data/alerts/integrated_alerts.json",
    "mongodb": {
        "enabled": True,
        "uri": "mongodb://localhost:27017",
        "db_name": "immigration_monitoring",
        "collection": "integrated_alerts"
    }
}

# ✅ Save Path Using Absolute Reference
config_dir = os.path.join(project_root, "src", "alert_integrator")
os.makedirs(config_dir, exist_ok=True)

config_path = os.path.join(config_dir, "alert_config.json")
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)

print("✅ alert_config.json created successfully at:", config_path)
