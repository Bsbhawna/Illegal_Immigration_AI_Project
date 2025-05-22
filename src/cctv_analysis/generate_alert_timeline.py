import pandas as pd
import matplotlib.pyplot as plt
import os

# Set paths
base_dir = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project"
log_csv_path = os.path.join(base_dir, r"data\logs\alert_log.csv")
timeline_chart_path = os.path.join(base_dir, r"data\final\alert_timeline.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(timeline_chart_path), exist_ok=True)

# Load alert log
try:
    df = pd.read_csv(log_csv_path, parse_dates=['Timestamp'])
except FileNotFoundError:
    print(f"❌ Log file not found at: {log_csv_path}")
    exit()
except Exception as e:
    print(f"⚠️ Error reading CSV: {e}")
    exit()

# Convert timestamp to minute-level granularity
df['Minute'] = df['Timestamp'].dt.floor('min')

# Count alerts per minute
alert_counts = df.groupby('Minute').size().reset_index(name='Alerts')

# Plot timeline chart
plt.figure(figsize=(12, 6))
plt.plot(alert_counts['Minute'], alert_counts['Alerts'], marker='o', linestyle='-', color='red')
plt.title('Alert Timeline - Persons Detected Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Alerts')
plt.grid(True)
plt.tight_layout()

# Save plot
try:
    plt.savefig(timeline_chart_path)
    print(f"✅ Timeline chart saved at: {timeline_chart_path}")
except Exception as e:
    print(f"❌ Failed to save timeline image: {e}")
finally:
    plt.close()
