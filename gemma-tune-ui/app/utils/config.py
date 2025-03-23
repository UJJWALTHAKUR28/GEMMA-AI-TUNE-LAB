import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

def get_api_key():
    """Fetches API key from environment variables"""
    return os.getenv("GOOGLE_AI_STUDIO_API_KEY")
