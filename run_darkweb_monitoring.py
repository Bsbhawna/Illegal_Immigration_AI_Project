import os
import sys
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta

# ------------------ CONFIG ------------------ #
# Define paths
project_root = os.path.abspath("C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project")
data_csv_path = os.path.join(project_root, "data", "synthetic", "immigration_darkweb_data.csv")
alerts_json_path = os.path.join(project_root, "data", "processed", "darkweb_alerts.json")
heatmap_plot_path = os.path.join(project_root, "src", "darkweb_monitoring", "plots", "dark_web_alerts_heatmap.png")
os.makedirs(os.path.dirname(heatmap_plot_path), exist_ok=True)

# MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["immigration_monitoring"]
collection = db["dark_web_alerts"]

# Add project root to path for module import
sys.path.append(project_root)

# ------------------ IMPORT YOUR FUNCTION ------------------ #
from src.darkweb_monitoring.data_monitoring import monitor_darkweb

# 🆕 Function to generate random date (last 20 days)
def generate_random_recent_date():
    days_ago = random.randint(0, 20)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

# ------------------ MAIN SCRIPT ------------------ #
def main():
    print("\n🔍 Starting Immigration Dark Web Monitoring System...\n")

    # Step 1: Run Monitoring
    alerts = monitor_darkweb(data_csv_path, alerts_json_path)

    if alerts:
        print(f"[✅ ALERTS GENERATED] Total Alerts: {len(alerts)}\n")

        # 🆕 Inject random date if missing
        for alert in alerts:
            if 'date' not in alert or not alert['date']:
                alert['date'] = generate_random_recent_date()

        for alert in alerts[:10]:  # Display top 10 alerts
            print(alert)
        print("\n🔑 Sample Alert Keys:", alerts[0].keys())

        # Step 2: Push Alerts to MongoDB
        collection.insert_many(alerts)
        print("\n🗂️ Dark Web alerts pushed to MongoDB.")

        # Step 3: Create Heatmap Visualization
        df_alerts = pd.DataFrame(alerts)
        df_alerts['date'] = pd.to_datetime(df_alerts['date'], errors='coerce').dt.date
        heatmap_data = df_alerts.groupby(['date', 'keyword']).size().unstack(fill_value=0)

        plt.figure(figsize=(12, 6))
        plt.title("🔥 Dark Web Alerts Heatmap (Keyword vs Date)")
        plt.xlabel("Date")
        plt.ylabel("Keyword")
        plt.imshow(heatmap_data.T, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Alert Count')
        plt.yticks(range(len(heatmap_data.columns)), heatmap_data.columns)
        plt.xticks(range(len(heatmap_data.index)), heatmap_data.index, rotation=45)
        plt.tight_layout()
        plt.savefig(heatmap_plot_path)
        plt.show()

        print(f"\n📊 Heatmap saved at: {heatmap_plot_path}")
    else:
        print("[INFO] No suspicious activity detected.")

# ------------------ ENTRY POINT ------------------ #
if __name__ == "__main__":
    main()
