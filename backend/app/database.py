import os
from pymongo import MongoClient

# O backend/.env é carregado em app/__init__.py, antes deste módulo.
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URL)
db = client[MONGO_DB]

users_collection = db["users"]
