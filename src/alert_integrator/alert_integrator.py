import os
import json
import glob
from pymongo import MongoClient


class AlertIntegrator:
    def __init__(self, config_path="alert_config.json"):
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Set root to project directory
        self.project_root = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project"
        os.chdir(self.project_root)
        print(f"[INFO] Working directory set to {os.getcwd()}")

        # Extract config items
        self.alert_paths = {
            "cctv": self.config.get("cctv_alerts", []),
            "twitter": self.config.get("twitter_alerts", []),
            "reddit": self.config.get("reddit_alerts", []),
            "darkweb": self.config.get("darkweb_alerts", [])
        }

        self.output_path = self.config.get("integrated_alerts_output", "data/alerts/integrated_alerts.json")

        # MongoDB
        mongo_cfg = self.config.get("mongodb", {})
        self.mongo_enabled = mongo_cfg.get("enabled", False)
        self.mongo_uri = mongo_cfg.get("uri", "")
        self.mongo_db = mongo_cfg.get("db_name", "")
        self.mongo_collection_name = mongo_cfg.get("collection", "")
        self.mongo_collection = None

        if self.mongo_enabled:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            self.mongo_collection = db[self.mongo_collection_name]

    def collect_all_alerts(self):
        all_alerts = []
        for source, paths in self.alert_paths.items():
            if not isinstance(paths, list):
                paths = [paths]

            for path in paths:
                full_path = os.path.join(self.project_root, path) if not os.path.isabs(path) else path
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_alerts.extend(data)
                            elif isinstance(data, dict):
                                all_alerts.append(data)
                            else:
                                print(f"[WARNING] Unknown format in {path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load {path}: {e}")
                else:
                    print(f"[WARNING] Missing: {full_path}")

        return all_alerts

    def save_to_json(self, alerts):
        out_path = os.path.join(self.project_root, self.output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(alerts, f, indent=4)

    def save_to_mongo(self, alerts):
        if self.mongo_enabled and alerts:
            self.mongo_collection.insert_many(alerts)
        elif not alerts:
            print("[INFO] No alerts to insert into MongoDB")

    def run(self):
        print("[INFO] Starting alert integration...")
        alerts = self.collect_all_alerts()
        print(f"[INFO] Collected {len(alerts)} total alerts.")

        self.save_to_json(alerts)
        print(f"[INFO] Saved to {self.output_path}")

        if self.mongo_enabled:
            self.save_to_mongo(alerts)
            print("[INFO] Inserted into MongoDB")

        print("[INFO] Alert integration complete.")


# ✅ Run in script or Jupyter
if __name__ == "__main__":
    integrator = AlertIntegrator()
    integrator.run()
