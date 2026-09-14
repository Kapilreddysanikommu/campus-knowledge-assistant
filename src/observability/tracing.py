"""
Langfuse tracing client used by the /query endpoint.

get_tracing_client() returns a singleton client configured from the
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST environment
variables (LANGFUSE_HOST defaults to Langfuse Cloud if unset).
"""

from dotenv import load_dotenv
from langfuse import Langfuse, get_client

load_dotenv()


def get_tracing_client() -> Langfuse:
    return get_client()
