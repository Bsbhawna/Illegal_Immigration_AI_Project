import os
import json
import pandas as pd
from datetime import datetime

# ✅ Project Paths
project_root = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project"
input_csv = os.path.join(project_root, "data", "final","daily_summary.csv")
output_json = os.path.join(project_root, "src","cctv_analysis", "dashboard_ready", "cctv_alerts.json")

# ✅ Load and Process
try:
    df = pd.read_csv(input_csv)

    alerts = []
    for idx, row in df.iterrows():
        alert = {
            "timestamp": row.get("timestamp", datetime.now().isoformat()),
            "frame": int(row.get("frame", idx)),  # fallback: index
            "location": row.get("location", "Indo-Bangladesh border"),
            "alert_type": "Border Intrusion",
            "person_count": int(row.get("person_count", 1)),
            "video_source": "final_alert_overlay.avi"
        }
        alerts.append(alert)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=4, ensure_ascii=False)

    print("✅ CCTV alert JSON created:", output_json)

except Exception as e:
    print("❌ Failed to generate CCTV alerts JSON:", e)
