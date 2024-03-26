from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("TEST"))
# This line brings all environment variables from .env into os.environ

