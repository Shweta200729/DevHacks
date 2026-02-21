import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt

load_dotenv(override=True)

url: str = os.environ.get("SUPABASE")
key: str = os.environ.get("SUPABASE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found in .env file")

supabase: Client = create_client(url, key)

from pydantic import BaseModel, EmailStr
from fastapi import FastAPI, HTTPException

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


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
        print(f"Signup error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Database error saving new user: {str(e)}"
        )


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    try:
        # Fetch user by email
        response = (
            supabase.table("users").select("*").eq("email", request.email).execute()
        )
        users = response.data

        if not users:
            raise HTTPException(status_code=400, detail="Invalid email or password")

        user = users[0]

        # Verify password using bcrypt
        if not bcrypt.checkpw(
            request.password.encode("utf-8"), user["password_hash"].encode("utf-8")
        ):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        # Return user details securely (without password_hash)
        return {
            "message": "Login successful",
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "phone": user["phone"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello World"}


# ── Mount the Federated Learning sub-app under /fl ──────────────────────────
# All dashboard API calls use the /fl/* prefix. Without this mount,
# every /fl/* request returns 404 from the root auth-only app.
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.abspath(_os.path.dirname(__file__)))

from server.main import app as fl_app  # noqa: E402

app.mount("/fl", fl_app)
