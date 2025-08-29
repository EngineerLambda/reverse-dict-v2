import google.genai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

class WordsParser(BaseModel):
    words: list[str] = Field(..., description="List of words that match the user's description")
    definitions: list[str] = Field(..., description="Definition of each word picked")

def get_words(description: str):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            {"role": "user", "parts": [
                {"text": f"{description}"}
            ]}
        ],
        config={
            "system_instruction": "Given a description, give out five words that match the user's description the closest along with their definitions",
            "response_mime_type": "application/json",
            "response_schema": WordsParser
        }
    )

    return response
