from dotenv import load_dotenv
import os

# Load .env file FIRST
load_dotenv()

# Now your other imports
from openai import OpenAI

# Rest of your code...
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))