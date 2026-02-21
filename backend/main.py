import os
from fastapi import FastAPI
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE")
key: str = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found in .env file")

supabase: Client = create_client(url, key)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
