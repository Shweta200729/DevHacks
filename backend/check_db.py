import os
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt

load_dotenv()
url = os.environ.get("SUPABASE")
key = os.environ.get("SUPABASE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

response = (
    supabase.table("users").select("*").eq("email", "testbylocal@example.com").execute()
)
users = response.data
print("Users in DB:", users)

if users:
    user = users[0]
    stored_hash = user["password_hash"]
    print("Stored hash:", stored_hash)

    password = "Password123!"
    is_valid = bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    print("Is valid:", is_valid)
