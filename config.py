import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


MODEL = "llama-3.3-70b-versatile"


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to the .env file."
    )