from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from paperdex.retrieval import Retriever

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)


llm = ChatGroq(model="openai/gpt-oss-120b")

retriever = Retriever()
