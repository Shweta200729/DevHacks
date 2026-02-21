/**
 * lib/api.ts
 *
 * Central API client for the Federated Learning backend.
 * All fetch calls in the dashboard should use these helpers so that:
 *  - The base URL is configured in one place.
 *  - Every call handles network errors gracefully.
 *  - Types are enforced at the boundary.
 */

const BASE = "http://localhost:8000/fl";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MetricsResponse {
    current_version: number;
    pending_queue_size: number;
    evaluations: EvalRow[];
    aggregations: AggRow[];
}

export interface EvalRow {
    id: string;
    version_id: number;
    accuracy: number;   // 0–1
    loss: number;
    created_at?: string;
}

export interface AggRow {
    id: string;
    version_id: number;
    method: string;
    total_accepted: number;
    total_rejected: number;
    created_at?: string;
}

export interface ClientRow {
    id: string;
    client_id: string;      // UUID
    status: "ACCEPT" | "REJECT";
    norm_value: number | null;
    distance_value: number | null;
    reason: string;
    created_at?: string;
}

export interface ModelVersion {
    id: string;
    version_num: number;
    file_path: string;
    created_at: string;
}

export interface AdminConfig {
    dp_enabled: boolean;
    dp_clip_norm: number;
    dp_noise_mult: number;
    min_update_queue: number;
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T | null> {
    try {
        const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
        if (!res.ok) return null;
        return res.json() as Promise<T>;
    } catch {
        return null;
    }
}

async function post<T>(path: string, body: unknown): Promise<T | null> {
    try {
        const res = await fetch(`${BASE}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!res.ok) return null;
        return res.json() as Promise<T>;
    } catch {
        return null;
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/** GET /fl/metrics — evaluations + aggregation history + queue size */
export async function fetchMetrics(): Promise<MetricsResponse | null> {
    return get<MetricsResponse>("/metrics");
}

/** GET /fl/clients — last 50 client update attempts */
export async function fetchClients(): Promise<ClientRow[]> {
    const res = await get<{ data: ClientRow[] }>("/clients");
    return res?.data ?? [];
}

/** GET /fl/versions — all stored model versions */
export async function fetchVersions(): Promise<ModelVersion[]> {
    const res = await get<{ data: ModelVersion[] }>("/versions");
    return res?.data ?? [];
}

/** GET /fl/admin/config */
export async function fetchConfig(): Promise<AdminConfig | null> {
    return get<AdminConfig>("/admin/config");
}

/** POST /fl/admin/config */
export async function saveConfig(cfg: AdminConfig): Promise<boolean> {
    const res = await post<{ status: string }>("/admin/config", cfg);
    return res?.status === "ok";
}

/** POST /fl/simulate */
export async function simulate(
    clientName: string,
    isMalicious: boolean,
    maliciousMultiplier = 50.0
): Promise<void> {
    await post("/simulate", {
        client_name: clientName,
        is_malicious: isMalicious,
        malicious_multiplier: maliciousMultiplier,
    });
}

/** Trigger model download */
export function getModelDownloadUrl(versionId?: string): string {
    return versionId
        ? `${BASE}/model/download?version_id=${versionId}`
        : `${BASE}/model/download`;
}

/** Poll helper: call cb every intervalMs ms, and immediately on mount */
export function startPolling(cb: () => void, intervalMs = 3000): () => void {
    cb();
    const id = setInterval(cb, intervalMs);
    return () => clearInterval(id);
}
