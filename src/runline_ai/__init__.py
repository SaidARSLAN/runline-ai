from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

from langchain_groq import ChatGroq  # noqa: E402
from paperdex.retrieval import Retriever  # noqa: E402

llm = ChatGroq(model="openai/gpt-oss-120b")

retriever = Retriever()
