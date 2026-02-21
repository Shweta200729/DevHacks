"use client";
import React, { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TerminalSquare, RefreshCw } from "lucide-react";
import { fetchMetrics, fetchClients, ClientRow, AggRow, startPolling } from "@/lib/api";

interface LogEntry {
    id: string;
    timestamp: string;  // ISO string
    type: "INFO" | "WARN" | "ERROR" | "SUCCESS";
    message: string;
    node: string;
}

function buildLogs(aggs: AggRow[], clients: ClientRow[]): LogEntry[] {
    const logs: LogEntry[] = [];

    // One log entry per aggregation round
    aggs.forEach(agg => {
        logs.push({
            id: `agg-${agg.id}`,
            timestamp: agg.created_at ?? new Date(0).toISOString(),
            type: "INFO",
            message: `Round v${agg.version_id} aggregated via ${agg.method}. ` +
                `Accepted: ${agg.total_accepted} | Rejected: ${agg.total_rejected}`,
            node: "Master Node",
        });
    });

    // One log entry per client update attempt
    clients.forEach(c => {
        const accepted = c.status === "ACCEPT";
        logs.push({
            id: `client-${c.id}`,
            timestamp: c.created_at ?? new Date(0).toISOString(),
            type: accepted ? "SUCCESS" : "WARN",
            message: accepted
                ? `Client update ACCEPTED. Norm: ${c.norm_value?.toFixed(4) ?? "—"}, ` +
                `Distance: ${c.distance_value?.toFixed(4) ?? "—"}`
                : `Client update REJECTED — ${c.reason}. ` +
                `Norm: ${c.norm_value?.toFixed(4) ?? "—"}, ` +
                `Distance: ${c.distance_value?.toFixed(4) ?? "—"}`,
            node: `Client ${c.client_id.slice(0, 8)}…`,
        });
    });

    // Newest first
    return logs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

const LOG_COLORS: Record<string, string> = {
    INFO: "text-blue-400",
    WARN: "text-yellow-400",
    ERROR: "text-red-400",
    SUCCESS: "text-green-400",
};

export default function LogsPage() {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isRefreshing, setRefreshing] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    const load = useCallback(async () => {
        setRefreshing(true);
        const [metrics, clients] = await Promise.all([fetchMetrics(), fetchClients()]);
        const built = buildLogs(metrics?.aggregations ?? [], clients);
        setLogs(built.length > 0 ? built : [{
            id: "init",
            timestamp: new Date().toISOString(),
            type: "INFO",
            message: "FL Aggregation Server initialised. Awaiting client connections…",
            node: "System",
        }]);
        setRefreshing(false);
    }, []);

    useEffect(() => startPolling(load, 5000), [load]);

    return (
        <div className="flex flex-col gap-8 pb-10 h-full">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-2">
                    <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">System Logs</h2>
                    <p className="text-slate-500">Immutable audit trail of client submissions and master node aggregations.</p>
                </div>
                <span className="text-xs text-slate-400">{logs.length} entries</span>
            </div>

            <Card className="bg-slate-950 border-slate-900 shadow-xl grow flex flex-col overflow-hidden">
                <CardHeader className="border-b border-slate-800 bg-slate-900 py-3 flex flex-row items-center justify-between">
                    <CardTitle className="text-sm font-mono text-slate-300 flex items-center gap-2">
                        <TerminalSquare className="w-4 h-4 text-slate-400" />
                        bash — root@fl-master-node
                    </CardTitle>
                    <button
                        onClick={load}
                        className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-slate-800"
                        title="Refresh logs"
                    >
                        <RefreshCw className={`w-4 h-4 ${isRefreshing ? "animate-spin" : ""}`} />
                    </button>
                </CardHeader>

                <CardContent className="p-0 overflow-y-auto h-[600px] font-mono text-sm bg-[#0C111D]">
                    <div className="p-4 flex flex-col gap-0.5 w-full">
                        {logs.map(log => (
                            <div key={log.id} className="py-1 border-b border-slate-800/50 hover:bg-slate-900/40 flex gap-4 w-full">
                                <span className="text-slate-600 shrink-0 select-none w-20">
                                    {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                                </span>
                                <span className={`font-bold w-16 shrink-0 ${LOG_COLORS[log.type] ?? "text-slate-300"}`}>
                                    [{log.type}]
                                </span>
                                <span className="text-slate-400 shrink-0 w-32 truncate">{log.node}</span>
                                <span className="text-slate-300 break-all grow">{log.message}</span>
                            </div>
                        ))}

                        <div ref={bottomRef} className="mt-4 flex items-center gap-2 animate-pulse text-slate-500">
                            <span className="w-2 h-4 bg-slate-400 inline-block" />
                            Listening for incoming connections…
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
