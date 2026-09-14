"""
Manages the PostgreSQL connection used by the storage layer.

Connection settings are read from environment variables so the same code
works against a local Docker container, a native install, or a hosted
database later, without any code changes. A .env file in the project root
is loaded automatically if present.
"""

import os

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PostgresConnection

load_dotenv()

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "5432"
DEFAULT_DATABASE = "campus_knowledge_assistant"
DEFAULT_USER = "postgres"
DEFAULT_PASSWORD = "postgres"


def get_connection() -> PostgresConnection:
    """Open a new connection to the PostgreSQL database.

    Ensures the pgvector extension is enabled and registers it with
    psycopg2, so that Python lists can be passed directly as values for
    VECTOR columns. The caller is responsible for closing the connection
    when done.
    """
    connection = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", DEFAULT_HOST),
        port=os.environ.get("POSTGRES_PORT", DEFAULT_PORT),
        dbname=os.environ.get("POSTGRES_DB", DEFAULT_DATABASE),
        user=os.environ.get("POSTGRES_USER", DEFAULT_USER),
        password=os.environ.get("POSTGRES_PASSWORD", DEFAULT_PASSWORD),
    )

    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.commit()

    register_vector(connection)
    return connection
