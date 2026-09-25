import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


MODEL = "openai/gpt-oss-120b"


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to the .env file."
    )