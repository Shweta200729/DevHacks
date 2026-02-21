"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TerminalSquare, RefreshCw } from "lucide-react";

interface LogEntry {
    id: string | number;
    timestamp: string;
    type: "INFO" | "WARN" | "ERROR" | "SUCCESS";
    message: string;
    node?: string;
}

export default function LogsPage() {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isRefreshing, setIsRefreshing] = useState(false);

    const fetchLogs = async () => {
        setIsRefreshing(true);
        try {
            // For now, we will construct a mock chronological log stream from 
            // the /metrics and /clients endpoints to simulate a real log server.
            // In a production app, we'd add an /api/logs route filtering Winston or Stdout logs.
            const [metricsRes, clientsRes] = await Promise.all([
<<<<<<< HEAD
                fetch("http://localhost:8000/metrics").catch(() => null),
                fetch("http://localhost:8000/clients").catch(() => null)
=======
                fetch("http://localhost:8000/fl/metrics").catch(() => null),
                fetch("http://localhost:8000/fl/clients").catch(() => null)
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
            ]);

            let mergedLogs: LogEntry[] = [];

            if (metricsRes?.ok) {
                const mJson = await metricsRes.json();
                mJson.aggregations?.forEach((agg: any) => {
                    mergedLogs.push({
                        id: `agg-${agg.id}`,
                        timestamp: agg.created_at || new Date().toISOString(),
                        type: "INFO",
                        message: `Aggregated Version ${agg.version_id} using ${agg.method}. Valid: ${agg.total_accepted}, Rejected: ${agg.total_rejected}`,
                        node: "Master Node"
                    });
                });
            }

            if (clientsRes?.ok) {
                const cJson = await clientsRes.json();
                cJson.data?.forEach((c: any) => {
                    mergedLogs.push({
                        id: `client-${c.id}`,
                        timestamp: c.created_at || new Date().toISOString(),
                        type: c.status === "ACCEPT" ? "SUCCESS" : "WARN",
                        message: c.status === "ACCEPT"
                            ? `Client update validated. Norm: ${c.norm_value?.toFixed(2)}`
                            : `Poisoning detected: ${c.reason}. Norm: ${c.norm_value?.toFixed(2)}, Dist: ${c.distance_value?.toFixed(2)}`,
                        node: c.client_id
                    });
                });
            }

            // Sort by timestamp descending
            mergedLogs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

            // Artificial insertion if empty (for UI demo purposes)
            if (mergedLogs.length === 0) {
                mergedLogs = [{
                    id: 'init',
                    timestamp: new Date().toISOString(),
                    type: 'INFO',
                    message: "FL Aggregator Server Initialized. Awaiting connections...",
                    node: "System"
                }];
            }

            setLogs(mergedLogs);
        } catch (e) {
            console.error(e);
        } finally {
            setIsRefreshing(false);
        }
    };

    useEffect(() => {
        fetchLogs();
        const interval = setInterval(fetchLogs, 5000); // Polling every 5s for logs
        return () => clearInterval(interval);
    }, []);

    const getLogColor = (type: string) => {
        switch (type) {
            case "INFO": return "text-blue-400";
            case "WARN": return "text-yellow-400";
            case "ERROR": return "text-red-400";
            case "SUCCESS": return "text-green-400";
            default: return "text-slate-300";
        }
    };

    return (
        <div className="flex flex-col gap-8 pb-10 h-full">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-2">
                    <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">System Logs</h2>
                    <p className="text-slate-500">Immutable audit trail of edge client submissions and master node aggregations.</p>
                </div>
            </div>

<<<<<<< HEAD
            <Card className="bg-slate-950 border-slate-900 shadow-xl flex-grow flex flex-col overflow-hidden">
=======
            <Card className="bg-slate-950 border-slate-900 shadow-xl grow flex flex-col overflow-hidden">
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
                <CardHeader className="border-b border-slate-800 bg-slate-900 py-3 flex flex-row items-center justify-between">
                    <CardTitle className="text-sm font-mono text-slate-300 flex items-center gap-2">
                        <TerminalSquare className="w-4 h-4 text-slate-400" />
                        bash - root@fl-master-node
                    </CardTitle>
                    <button
                        onClick={fetchLogs}
                        className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-slate-800"
                    >
                        <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                    </button>
                </CardHeader>
                <CardContent className="p-0 overflow-y-auto h-[600px] font-mono text-sm bg-[#0C111D]">
                    <div className="p-4 flex flex-col content-start w-full">
                        {logs.map((log) => (
                            <div key={log.id} className="py-1 border-b border-slate-800/50 hover:bg-slate-900/50 flex gap-4 w-full">
                                <span className="text-slate-600 shrink-0 select-none">
                                    {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                </span>
                                <span className={`font-semibold w-16 shrink-0 ${getLogColor(log.type)}`}>
                                    [{log.type}]
                                </span>
                                {log.node && (
                                    <span className="text-slate-400 shrink-0 w-28 truncate">
                                        {log.node}
                                    </span>
                                )}
<<<<<<< HEAD
                                <span className="text-slate-300 break-words flex-grow">
=======
                                <span className="text-slate-300 text-wrap break-words grow">
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
                                    {log.message}
                                </span>
                            </div>
                        ))}
                        {logs.length === 0 && !isRefreshing && (
                            <div className="text-slate-600 italic mt-4">No system logs available.</div>
                        )}
                        <div className="mt-4 flex items-center gap-2 animate-pulse text-slate-500">
                            <span className="w-2 h-4 bg-slate-400 inline-block" />
                            Listening for incoming connections...
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
