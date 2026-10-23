from pathlib import Path
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

ENV_FILE = Path(__file__).resolve().parent / "app.env"
SETTINGS = dotenv_values(ENV_FILE)


def get_llm(temperature=0.2):
    return ChatOpenAI(
        model="openrouter/free",
        api_key=SETTINGS["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        # free models stall often; without this one dead provider blocks the app
        timeout=45,
        max_retries=1,
    )
