import os
import torch
import logging
from supabase import create_client, Client
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SupabaseManager:
    """Handles persistence to Supabase and tracks models.

    All write methods are best-effort: they log errors but never raise,
    so the FL pipeline works even when Supabase is temporarily unreachable.

    All read methods return empty lists on failure so callers can fall
    back to in-memory accumulators transparently.
    """

    def __init__(self, base_dir: str = "fl_simulation_data"):
        self.base_dir  = base_dir
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        url: str = os.getenv("SUPABASE")
        key: str = os.getenv("SUPABASE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("Supabase credentials not found for SupabaseManager initialization.")
        self.supabase: Client = create_client(url, key)

        # Cache: client_name → DB UUID  (avoids repeated upserts)
        self._client_id_cache: Dict[str, str] = {}

    # ── Client registry ───────────────────────────────────────────────────────

    def get_or_create_client_db_id(self, client_name: str) -> Optional[str]:
        """Upsert a client row and return its UUID.

        Used so client_updates FK inserts succeed.  Result is cached in-process.
        Returns None on failure (caller should skip DB write, use in-memory path).
        """
        if client_name in self._client_id_cache:
            return self._client_id_cache[client_name]
        try:
            # Try to find existing row first
            res = (
                self.supabase.table("clients")
                .select("id")
                .eq("client_name", client_name)
                .limit(1)
                .execute()
            )
            if res.data:
                uid = res.data[0]["id"]
            else:
                ins = self.supabase.table("clients").insert({"client_name": client_name}).execute()
                uid = ins.data[0]["id"]
            self._client_id_cache[client_name] = uid
            return uid
        except Exception as e:
            logger.warning(f"[Storage] get_or_create_client_db_id({client_name}): {e}")
            return None

    def register_client(self, name: str) -> str:
        """Registers a mock edge client in DB and returns UUID."""
        return self.get_or_create_client_db_id(name) or ""

    # ── Model versions ────────────────────────────────────────────────────────

    def save_model_version(self, model_state_dict: Dict[str, torch.Tensor], version_num: int) -> int:
        """Saves local PT checkpoint and upserts a Supabase log row.
        Uses UPSERT so re-running with the same version_num (e.g. after a restart)
        updates the file_path instead of crashing with a unique-constraint error.
        Returns the internal DB row id.
        """
        file_path = os.path.join(self.models_dir, f"global_model_round_{version_num}.pt")
        torch.save(model_state_dict, file_path)

        try:
            # Upsert: if version_num already exists, update file_path + created_at
            res = (
                self.supabase.table("model_versions")
                .upsert(
                    {"version_num": version_num, "file_path": file_path},
                    on_conflict="version_num",
                )
                .execute()
            )
            return res.data[0]["id"]
        except Exception as e:
            logger.error(f"[Storage] save_model_version upsert failed: {e}")
            raise  # re-raise so caller knows — DO NOT swallow this

    def get_latest_version(self) -> Dict[str, Any]:
        """Gets metadata for the active highest version available."""
        res = (
            self.supabase.table("model_versions")
            .select("*")
            .order("version_num", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return {"version_num": 0, "file_path": None, "id": None}
        return res.data[0]

    # ── Write helpers ─────────────────────────────────────────────────────────

    def log_client_update(
        self,
        version_id: int,
        client_name: str,
        status: str,
        norm: float,
        dist: float,
        reason: str,
    ):
        """Write a client update row.  Upserts client first to satisfy FK."""
        client_db_id = self.get_or_create_client_db_id(client_name)
        if not client_db_id:
            logger.warning(f"[Storage] Skipping client_updates DB write — no UUID for {client_name}")
            return
        try:
            self.supabase.table("client_updates").insert({
                "version_id":     version_id,
                "client_id":      client_db_id,
                "status":         status,
                "norm_value":     round(norm, 4) if norm is not None else None,
                "distance_value": round(dist, 4) if dist is not None else None,
                "reason":         reason,
            }).execute()
        except Exception as e:
            logger.warning(f"[Storage] log_client_update failed: {e}")

    def log_aggregation(self, version_id: int, accepted: int, rejected: int, method: str):
        try:
            self.supabase.table("aggregation_logs").insert({
                "version_id":     version_id,
                "total_accepted": accepted,
                "total_rejected": rejected,
                "method":         method,
            }).execute()
        except Exception as e:
            logger.warning(f"[Storage] log_aggregation failed: {e}")

    def log_evaluation(self, version_id: int, loss: float, accuracy: float):
        try:
            self.supabase.table("evaluation_metrics").insert({
                "version_id": version_id,
                "loss":       round(loss, 6) if loss is not None else None,
                "accuracy":   round(accuracy, 6) if accuracy is not None else None,
            }).execute()
        except Exception as e:
            logger.warning(f"[Storage] log_evaluation failed: {e}")

    def log_train_epoch(
        self,
        client_id: str,
        round_num: int,
        epoch: int,
        loss: float,
        accuracy: float,
    ):
        """Persist a single per-epoch training metric row to train_history."""
        try:
            self.supabase.table("train_history").insert({
                "client_id": client_id,
                "round":     round_num,
                "epoch":     epoch,
                "loss":      round(loss, 6) if loss is not None else None,
                "accuracy":  round(accuracy, 6) if accuracy is not None else None,
            }).execute()
        except Exception as e:
            logger.warning(f"[Storage] log_train_epoch failed: {e}")

    # ── Bulk-read helpers (called during startup warm-up) ─────────────────────

    def read_recent_evaluations(self, limit: int = 50) -> List[Dict]:
        try:
            rows = (
                self.supabase.table("evaluation_metrics")
                .select("id, version_id, loss, accuracy, created_at")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(reversed(rows.data or []))
        except Exception as e:
            logger.warning(f"[Storage] read_recent_evaluations: {e}")
            return []

    def read_recent_aggregations(self, limit: int = 50) -> List[Dict]:
        try:
            rows = (
                self.supabase.table("aggregation_logs")
                .select("id, version_id, total_accepted, total_rejected, method, created_at")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(reversed(rows.data or []))
        except Exception as e:
            logger.warning(f"[Storage] read_recent_aggregations: {e}")
            return []

    def read_recent_client_updates(self, limit: int = 100) -> List[Dict]:
        try:
            rows = (
                self.supabase.table("client_updates")
                .select("id, version_id, client_id, status, norm_value, distance_value, reason, created_at")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            # Resolve client UUID → name using cached map (reverse lookup)
            uid_to_name = {v: k for k, v in self._client_id_cache.items()}
            result = []
            for row in reversed(rows.data or []):
                row = dict(row)
                uid = row.get("client_id", "")
                row["client_id"] = uid_to_name.get(uid, uid)  # prefer name, fall back to UUID
                result.append(row)
            return result
        except Exception as e:
            logger.warning(f"[Storage] read_recent_client_updates: {e}")
            return []

    def read_recent_versions(self, limit: int = 50) -> List[Dict]:
        try:
            rows = (
                self.supabase.table("model_versions")
                .select("id, version_num, file_path, created_at")
                .order("version_num", desc=True)
                .limit(limit)
                .execute()
            )
            return list(reversed(rows.data or []))
        except Exception as e:
            logger.warning(f"[Storage] read_recent_versions: {e}")
            return []

    def read_recent_train_history(self, limit: int = 300) -> List[Dict]:
        try:
            rows = (
                self.supabase.table("train_history")
                .select("id, client_id, round, epoch, loss, accuracy, created_at")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(reversed(rows.data or []))
        except Exception as e:
            logger.warning(f"[Storage] read_recent_train_history: {e}")
            return []
