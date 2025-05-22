import pymongo

def load_alerts():
    """Load alerts from MongoDB."""
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["immigration_monitoring"]
    collection = db["alerts"]
    alerts = list(collection.find({}, {"_id": 0}))
    return alerts
