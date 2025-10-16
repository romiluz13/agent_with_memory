from dotenv import load_dotenv
import os

# Load .env file FIRST
load_dotenv()

# Now your other imports
from pymongo import MongoClient

# Rest of your code...
uri = os.getenv("MONGODB_URI")