import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

LLM_API_KEY = os.getenv("GOOGLE_API_KEY")