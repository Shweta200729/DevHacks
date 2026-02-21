import os
import torch
from supabase import create_client, Client
from typing import Dict, Any

class SupabaseManager:
    """Handles strictly structured persistence to Supabase and tracking models."""
    def __init__(self, base_dir: str = "fl_simulation_data"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        url: str = os.getenv("SUPABASE")
        key: str = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("Supabase credentials not found for SupabaseManager initialization.")
        self.supabase: Client = create_client(url, key)

    def save_model_version(self, model_state_dict: Dict[str, torch.Tensor], version_num: int) -> int:
        """Saves local PT checkpoint and Supabase log. Returns internal DB Version ID."""
        file_path = os.path.join(self.models_dir, f"global_model_round_{version_num}.pt")
        torch.save(model_state_dict, file_path)
        
        res = self.supabase.table("model_versions").insert({
            "version_num": version_num,
            "file_path": file_path
        }).execute()
        return res.data[0]["id"]

    def get_latest_version(self) -> Dict[str, Any]:
        """Gets metadata for the active highest version available."""
        res = self.supabase.table("model_versions").select("*").order("version_num", desc=True).limit(1).execute()
        if len(res.data) == 0:
            return {"version_num": 0, "file_path": None, "id": None}
        return res.data[0]

    def log_client_update(self, version_id: int, client_id: str, status: str, norm: float, dist: float, reason: str):
        self.supabase.table("client_updates").insert({
            "version_id": version_id,
            "client_id": client_id,
            "status": status,
            "norm_value": round(norm, 4) if norm else None,
            "distance_value": round(dist, 4) if dist else None,
            "reason": reason
        }).execute()

    def log_aggregation(self, version_id: int, accepted: int, rejected: int, method: str):
        self.supabase.table("aggregation_logs").insert({
            "version_id": version_id,
            "total_accepted": accepted,
            "total_rejected": rejected,
            "method": method
        }).execute()

    def log_evaluation(self, version_id: int, loss: float, accuracy: float):
        self.supabase.table("evaluation_metrics").insert({
            "version_id": version_id,
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 4)
        }).execute()

    def register_client(self, name: str) -> str:
        """Registers a mock edge client in DB and returns UUID."""
        res = self.supabase.table("clients").insert({"client_name": name}).execute()
        return res.data[0]["id"]
