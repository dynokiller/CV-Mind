import os
from pymongo import MongoClient
from dotenv import load_dotenv
# Load environment variables from .env file

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME")

if not MONGO_URI or not MONGO_DBNAME:
    raise ValueError("MONGO_URI or MONGO_DBNAME not found in .env file")

client = MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]

# MongoDB connection
client = MongoClient(MONGO_URI)

db = client[MONGO_DBNAME]

# Collections
user_collection = db["user"]
stats_collection = db["stats"]
activity_collection = db["activity"]
file_integrity_collection = db["file_integrity"]
reset_tokens_collection = db["password_reset_tokens"]  

reset_tokens_collection = db["password_reset_tokens"]

# Auto delete expired password reset tokens
reset_tokens_collection.create_index(
    "expires_at",
    expireAfterSeconds=0
)