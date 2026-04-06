import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DBNAME = os.getenv("MONGO_DBNAME", "cv_mind")

# MongoDB connection
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.server_info()  # Force connection check
    print("MongoDB connected successfully.")
    db = client[MONGO_DBNAME]
    _mongo_available = True
except Exception as e:
    print(f"WARNING: MongoDB not available. Using TEMPORARY IN-MEMORY Database (mongomock). Error: {e}")
    import mongomock
    client = mongomock.MongoClient()
    db = client[MONGO_DBNAME]
    _mongo_available = False

user_collection = db["user"]
stats_collection = db["stats"]
activity_collection = db["activity"]
file_integrity_collection = db["file_integrity"]
reset_tokens_collection = db["password_reset_tokens"]

try:
    reset_tokens_collection.create_index(
        "expires_at",
        expireAfterSeconds=0
    )
except Exception as e:
    print(f"WARNING: Could not create TTL index (mongomock fallback): {e}")