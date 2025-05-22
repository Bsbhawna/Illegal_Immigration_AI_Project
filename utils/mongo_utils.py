# utils/mongo_utils.py

from pymongo import MongoClient

def get_mongo_collection(db_name, collection_name, uri="mongodb://localhost:27017/"):
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]
