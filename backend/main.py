import os
from fastapi import FastAPI
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt

load_dotenv()

url: str = os.environ.get("SUPABASE")
key: str = os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found in .env file")

supabase: Client = create_client(url, key)

from pydantic import BaseModel, EmailStr
from fastapi import FastAPI, HTTPException

app = FastAPI()


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    password: str
    confirm_password: str


@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    try:
        # Hash the password using bcrypt
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(request.password.encode("utf-8"), salt).decode(
            "utf-8"
        )

        # Insert into the Custom SQL 'users' table
        user_data = {
            "name": request.name,
            "email": request.email,
            "phone": request.phone,
            "password_hash": hashed_password,
        }
        response = supabase.table("users").insert(user_data).execute()

        return {"message": "User created successfully", "user": response.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello World"}
