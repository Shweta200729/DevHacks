"use client";
import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BorderBeam } from "@/components/ui/border-beam";
import { Activity, Network, ShieldCheck, Cpu, ArrowUpRight, CopyCheck, Wifi } from "lucide-react";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer
} from "recharts";
import {
    fetchMetrics, MetricsResponse, EvalRow, AggRow, startPolling
} from "@/lib/api";

export default function OverviewPage() {
    const [data, setData] = useState<MetricsResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    const load = useCallback(async () => {
        const json = await fetchMetrics();
        if (json) {
            // chronological order for charts (oldest → newest)
            json.evaluations = [...json.evaluations].reverse();
            json.aggregations = [...json.aggregations].reverse();
            setData(json);
            setLastUpdated(new Date());
        }
        setLoading(false);
    }, []);

    useEffect(() => startPolling(load, 3000), [load]);

    // ── derived values ─────────────────────────────────────────────────────────
    const totalAccepted = data?.aggregations.reduce((s, a) => s + a.total_accepted, 0) ?? 0;
    const totalRejected = data?.aggregations.reduce((s, a) => s + a.total_rejected, 0) ?? 0;
    const latestEval = data?.evaluations.at(-1) ?? null;
    const latestAgg = data?.aggregations.at(-1) ?? null;

    const kpis = [
        {
            title: "Model Version",
            value: `v${data?.current_version ?? 0}`,
            icon: Network,
            trend: "Active global checkpoint",
        },
        {
            title: "Aggregation Method",
            value: latestAgg?.method ?? "—",
            icon: Cpu,
            trend: "DP noise layer active",
        },
        {
            title: "Valid Updates",
            value: totalAccepted.toString(),
            icon: Activity,
            trend: `Queue: ${data?.pending_queue_size ?? 0} pending`,
        },
        {
            title: "Rejected (Byzantine)",
            value: totalRejected.toString(),
            icon: ShieldCheck,
            trend: "Malicious payloads blocked",
            highlight: true,
        },
        {
            title: "Latest Accuracy",
            value: latestEval ? `${(latestEval.accuracy * 100).toFixed(2)}%` : "—",
            icon: CopyCheck,
            trend: "Validated on MNIST test set",
            highlightKey: true,
        },
    ];

    // ── activity feed from real aggregation rows ───────────────────────────────
    const activityRows = [...(data?.aggregations ?? [])].reverse().slice(0, 6).map(agg => ({
        event: "Global Model Aggregated",
        details: `v${agg.version_id} via ${agg.method} — ${agg.total_accepted} accepted, ${agg.total_rejected} rejected`,
        status: "Success",
        time: agg.created_at ? new Date(agg.created_at).toLocaleTimeString() : `Round ${agg.version_id}`,
    }));

    return (
        <div className="flex flex-col gap-8 pb-10">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-2">
                    <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
                        Global System Overview
                    </h2>
                    <p className="text-slate-500">
                        Real-time metrics from your federated learning infrastructure.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                    <Wifi className={`w-3.5 h-3.5 ${data ? "text-green-500" : "text-slate-300"}`} />
                    {lastUpdated
                        ? `Updated ${lastUpdated.toLocaleTimeString()}`
                        : loading ? "Connecting…" : "No data"}
                </div>
            </div>

            {/* KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {kpis.map((kpi, i) => (
                    <Card
                        key={i}
                        className={`relative overflow-hidden transition-all duration-300 hover:shadow-md hover:-translate-y-0.5
                            ${kpi.highlightKey ? "border-blue-200 bg-blue-50/30"
                                : kpi.highlight ? "border-red-100 bg-red-50/20"
                                    : "bg-white border-slate-200"}`}
                    >
                        {kpi.highlightKey && <BorderBeam duration={8} size={150} />}
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium text-slate-500">{kpi.title}</CardTitle>
                            <kpi.icon className={`h-4 w-4 ${kpi.highlightKey ? "text-blue-600" : kpi.highlight ? "text-red-500" : "text-slate-400"}`} />
                        </CardHeader>
                        <CardContent>
                            <div className={`text-2xl font-bold ${kpi.highlightKey ? "text-blue-700" : kpi.highlight ? "text-red-600" : "text-slate-900"}`}>
                                {kpi.value}
                            </div>
                            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                {i === 2 && <ArrowUpRight className="h-3 w-3 text-green-500" />}
                                {kpi.trend}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Convergence (Accuracy per Round)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                            {(data?.evaluations.length ?? 0) > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={data!.evaluations}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                        <XAxis dataKey="version_id" stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={t => `v${t}`} />
                                        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} tickFormatter={t => `${(t * 100).toFixed(0)}%`} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: "#fff", borderColor: "#e2e8f0", borderRadius: "8px" }}
                                            formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, "Accuracy"]}
                                        />
                                        <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={3}
                                            dot={{ fill: "#2563eb", r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex h-full items-center justify-center text-slate-400 text-sm">
                                    No evaluation data yet — run a simulation or upload a dataset.
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Global Training Loss</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                            {(data?.evaluations.length ?? 0) > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={data!.evaluations}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                        <XAxis dataKey="version_id" stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={t => `v${t}`} />
                                        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: "#fff", borderColor: "#e2e8f0", borderRadius: "8px" }}
                                            formatter={(v: number) => [v.toFixed(4), "Loss"]}
                                        />
                                        <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={3}
                                            dot={{ fill: "#ef4444", r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex h-full items-center justify-center text-slate-400 text-sm">
                                    No evaluation data yet.
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Activity Feed */}
            <Card className="bg-white border-slate-200 shadow-sm mt-4">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Recent Aggregation Activity</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-50/50">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-100/50 text-slate-500 uppercase text-xs">
                                <tr>
                                    <th className="px-6 py-3 font-semibold">Event</th>
                                    <th className="px-6 py-3 font-semibold">Details</th>
                                    <th className="px-6 py-3 font-semibold">Status</th>
                                    <th className="px-6 py-3 font-semibold text-right">Time</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200">
                                {activityRows.length > 0 ? activityRows.map((row, i) => (
                                    <tr key={i} className="bg-white hover:bg-slate-50 transition-colors">
                                        <td className="px-6 py-4 font-medium text-slate-900">{row.event}</td>
                                        <td className="px-6 py-4 text-slate-500 text-xs">{row.details}</td>
                                        <td className="px-6 py-4">
                                            <span className="px-2.5 py-1 rounded-full text-xs font-medium border bg-green-50 text-green-700 border-green-200">
                                                {row.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right text-slate-400 text-xs">{row.time}</td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan={4} className="px-6 py-8 text-center text-slate-500">
                                            No aggregation events yet. Fire up some simulated edge nodes on the Clients page!
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
