"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BorderBeam } from "@/components/ui/border-beam";
import {
    Activity, Network, ShieldCheck, Cpu, ArrowUpRight, CopyCheck,
    AlertTriangle, TrendingDown, TrendingUp, Zap, Clock, CheckCircle2
} from "lucide-react";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend, Area, AreaChart, ComposedChart, Bar
} from "recharts";

interface MetricsData {
    current_version: number;
    evaluations: any[];
    aggregations: any[];
    pending_queue_size: number;
}

export default function OverviewPage() {
    const [data, setData] = useState<MetricsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    const fetchMetrics = async () => {
        try {
            const res = await fetch("http://localhost:8000/fl/metrics");
            if (res.ok) {
                const json = await res.json();
                json.evaluations = [...json.evaluations].reverse();
                json.aggregations = [...json.aggregations].reverse();
                setData(json);
                setLastUpdated(new Date());
            }
        } catch (e) {
            console.error("Failed fetching metrics", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 3000);
        return () => clearInterval(interval);
    }, []);

    const calculateTotalClients = () => {
        if (!data?.aggregations) return { acc: 0, rej: 0 };
        let acc = 0, rej = 0;
        data.aggregations.forEach(a => { acc += a.total_accepted; rej += a.total_rejected; });
        return { acc, rej };
    };

    const stats = calculateTotalClients();
    const latestEval = data?.evaluations?.length ? data.evaluations[data.evaluations.length - 1] : null;
    const prevEval = data?.evaluations?.length && data.evaluations.length > 1 ? data.evaluations[data.evaluations.length - 2] : null;
    const latestAgg = data?.aggregations?.length ? data.aggregations[data.aggregations.length - 1] : null;

    const accDelta = latestEval && prevEval ? ((latestEval.accuracy - prevEval.accuracy) * 100) : null;
    const lossDelta = latestEval && prevEval ? (latestEval.loss - prevEval.loss) : null;

    const kpis = [
        { title: "Model Version", value: `v${data?.current_version || 0}`, icon: Network, trend: "Live Synced" },
        { title: "Aggregation Method", value: latestAgg?.method || "—", icon: Cpu, trend: "DP layer active" },
        { title: "Accepted Updates", value: stats.acc.toString(), icon: Activity, trend: `Queue: ${data?.pending_queue_size || 0} pending` },
        { title: "Blocked / Rejected", value: stats.rej.toString(), icon: ShieldCheck, trend: "Byzantine payloads blocked", highlight: true },
        { title: "Latest Accuracy", value: latestEval ? `${(latestEval.accuracy * 100).toFixed(2)}%` : "N/A", icon: CopyCheck, trend: accDelta !== null ? `${accDelta >= 0 ? "↑" : "↓"} ${Math.abs(accDelta).toFixed(2)}% this round` : "Awaiting data", highlightKey: true },
    ];

    // Combined chart: loss + accuracy on dual axes
    const combinedChartData = (data?.evaluations ?? []).map(e => ({
        version: `v${e.version_id}`,
        accuracy: parseFloat((e.accuracy * 100).toFixed(2)),
        loss: parseFloat(e.loss.toFixed(4)),
    }));

    // Client acceptance rate per round
    const acceptanceData = (data?.aggregations ?? []).map(a => ({
        round: `v${a.version_id}`,
        accepted: a.total_accepted,
        rejected: a.total_rejected,
    }));

    // Real-time insights derived from live data
    const insights: { icon: any; color: string; label: string; detail: string; type: "ok" | "warn" | "info" }[] = [];

    if (latestEval) {
        const acc = latestEval.accuracy * 100;
        if (acc >= 80) {
            insights.push({ icon: CheckCircle2, color: "text-green-500", label: "Model Converging Well", detail: `${acc.toFixed(1)}% accuracy on validation set`, type: "ok" });
        } else if (acc >= 50) {
            insights.push({ icon: TrendingUp, color: "text-yellow-500", label: "Model Still Learning", detail: `${acc.toFixed(1)}% accuracy — more training rounds recommended`, type: "warn" });
        } else {
            insights.push({ icon: AlertTriangle, color: "text-red-500", label: "Underfitting Detected", detail: `Only ${acc.toFixed(1)}% accuracy — check dataset balance`, type: "warn" });
        }

        if (lossDelta !== null) {
            if (lossDelta < 0) {
                insights.push({ icon: TrendingDown, color: "text-green-500", label: "Loss Decreasing", detail: `Δ loss = ${lossDelta.toFixed(4)} — healthy gradient descent`, type: "ok" });
            } else if (lossDelta > 0.5) {
                insights.push({ icon: AlertTriangle, color: "text-red-500", label: "Loss Spiked", detail: `Δ loss = +${lossDelta.toFixed(4)} — possible poisoning or divergence`, type: "warn" });
            } else {
                insights.push({ icon: TrendingUp, color: "text-yellow-500", label: "Loss Stable / Plateauing", detail: `Δ loss = +${lossDelta.toFixed(4)} — consider adjusting LR`, type: "info" });
            }
        }
    }

    if (stats.rej > 0) {
        const rejectRate = stats.acc > 0 ? ((stats.rej / (stats.acc + stats.rej)) * 100).toFixed(1) : "100";
        insights.push({ icon: ShieldCheck, color: "text-indigo-500", label: "Byzantine Defense Active", detail: `${rejectRate}% of updates blocked by L2 anomaly detector`, type: "info" });
    }

    if (data?.pending_queue_size && data.pending_queue_size > 0) {
        insights.push({ icon: Clock, color: "text-blue-500", label: "Aggregation Queued", detail: `${data.pending_queue_size} updates pending — aggregation will trigger soon`, type: "info" });
    }

    if (data?.current_version && data.current_version > 0 && insights.length === 0) {
        insights.push({ icon: Zap, color: "text-slate-400", label: "System Idle", detail: "All rounds complete. Upload a dataset or fire a simulation to continue.", type: "info" });
    }

    if (!data && !loading) {
        insights.push({ icon: AlertTriangle, color: "text-red-500", label: "Cannot Reach Backend", detail: "Check that uvicorn is running on port 8000.", type: "warn" });
    }

    // Activity feed
    const activityFeed: any[] = [];
    if (data?.aggregations) {
        [...data.aggregations].reverse().slice(0, 5).forEach((agg: any) => {
            activityFeed.push({
                event: "Global Model Updated",
                details: `v${agg.version_id} via ${agg.method} — ${agg.total_accepted} valid, ${agg.total_rejected} blocked`,
                status: "Success",
                time: agg.created_at ? new Date(agg.created_at).toLocaleTimeString() : `Round ${agg.version_id}`
            });
        });
    }

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                    <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Global System Overview</h2>
                    <p className="text-slate-500 text-sm">
                        Real-time federated learning metrics.
                        {lastUpdated && <span className="ml-2 text-slate-400">Last updated: {lastUpdated.toLocaleTimeString()}</span>}
                    </p>
                </div>
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border ${data ? "bg-green-50 text-green-700 border-green-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                    <span className={`w-2 h-2 rounded-full ${data ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
                    {data ? "Server Live" : "Disconnected"}
                </div>
            </div>

            {/* ── KPI Cards ─────────────────────────────────────── */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {kpis.map((kpi, i) => (
                    <Card key={i} className={`relative overflow-hidden transition-all duration-300 hover:shadow-md hover:-translate-y-0.5 ${kpi.highlightKey ? "border-blue-200 bg-blue-50/30" : "bg-white border-slate-200"}`}>
                        {kpi.highlightKey && <BorderBeam duration={8} size={150} />}
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium text-slate-500">{kpi.title}</CardTitle>
                            <kpi.icon className={`h-4 w-4 ${kpi.highlightKey ? "text-blue-600" : kpi.highlight ? "text-red-500" : "text-slate-400"}`} />
                        </CardHeader>
                        <CardContent>
                            <div className={`text-2xl font-bold ${kpi.highlightKey ? "text-blue-700" : kpi.highlight ? "text-red-600" : "text-slate-900"}`}>{kpi.value}</div>
                            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                {i === 2 || i === 4 ? <ArrowUpRight className="h-3 w-3 text-green-500" /> : null}
                                {kpi.trend}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* ── Real-Time Insights Panel ─────────────────────────────────── */}
            <Card className="bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700 shadow-xl text-white">
                <CardHeader>
                    <CardTitle className="text-base font-bold text-white flex items-center gap-2">
                        <Zap className="w-4 h-4 text-yellow-400" />
                        Real-Time System Insights
                        <span className="ml-auto text-[10px] font-normal text-slate-400 animate-pulse">● Live</span>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {insights.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {insights.map((ins, i) => (
                                <div key={i} className={`flex items-start gap-3 p-3 rounded-lg border ${ins.type === "ok" ? "bg-green-900/20 border-green-700/40" : ins.type === "warn" ? "bg-red-900/20 border-red-700/40" : "bg-slate-700/40 border-slate-600/40"}`}>
                                    <ins.icon className={`w-5 h-5 mt-0.5 shrink-0 ${ins.color}`} />
                                    <div>
                                        <p className="text-sm font-semibold text-white">{ins.label}</p>
                                        <p className="text-xs text-slate-300 mt-0.5">{ins.detail}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-sm text-slate-400 italic">Loading insights...</div>
                    )}
                </CardContent>
            </Card>

            {/* ── Combined Accuracy + Loss Chart (Dual Axis) ─────────────────────────────────── */}
            <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Convergence — Accuracy & Loss Per Round</CardTitle>
                    <p className="text-sm text-slate-400 mt-0.5">Blue = Accuracy (left axis) · Red = Training Loss (right axis)</p>
                </CardHeader>
                <CardContent>
                    <div className="h-72 w-full rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                        {combinedChartData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={combinedChartData}>
                                    <defs>
                                        <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15} />
                                            <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                    <XAxis dataKey="version" stroke="#94a3b8" fontSize={12} tickLine={false} />
                                    <YAxis yAxisId="acc" stroke="#2563eb" fontSize={11} tickLine={false} axisLine={false}
                                        tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                                    <YAxis yAxisId="loss" orientation="right" stroke="#ef4444" fontSize={11}
                                        tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#fff", borderColor: "#e2e8f0", borderRadius: 8, boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }}
                                        formatter={(value: any, name: string) =>
                                            name === "accuracy" ? [`${value}%`, "Accuracy"] : [value, "Loss"]
                                        }
                                    />
                                    <Legend wrapperStyle={{ fontSize: 12 }} />
                                    <Area yAxisId="acc" type="monotone" dataKey="accuracy" name="accuracy"
                                        stroke="#2563eb" strokeWidth={2.5} fill="url(#accGrad)"
                                        dot={{ fill: "#2563eb", r: 3 }} activeDot={{ r: 5 }} />
                                    <Line yAxisId="loss" type="monotone" dataKey="loss" name="loss"
                                        stroke="#ef4444" strokeWidth={2} strokeDasharray="5 3"
                                        dot={{ fill: "#ef4444", r: 3 }} activeDot={{ r: 5 }} />
                                </ComposedChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-slate-400 text-sm">
                                No evaluation data yet. Upload a dataset or fire a simulation.
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* ── Acceptance Rate Bar Chart ──────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm">
                    <CardHeader>
                        <CardTitle className="text-base font-bold text-slate-900">Client Update Acceptance per Round</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-56 rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                            {acceptanceData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={acceptanceData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                        <XAxis dataKey="round" stroke="#94a3b8" fontSize={11} tickLine={false} />
                                        <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                                        <Tooltip contentStyle={{ backgroundColor: "#fff", borderColor: "#e2e8f0", borderRadius: 8 }} />
                                        <Legend wrapperStyle={{ fontSize: 12 }} />
                                        <Bar dataKey="accepted" name="Accepted" fill="#22c55e" radius={[3, 3, 0, 0]} />
                                        <Bar dataKey="rejected" name="Rejected" fill="#ef4444" radius={[3, 3, 0, 0]} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-slate-400 text-sm">No round data yet.</div>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* ── Recent Aggregation Activity ──────────────────────────────── */}
                <Card className="bg-white border-slate-200 shadow-sm">
                    <CardHeader>
                        <CardTitle className="text-base font-bold text-slate-900">Recent Aggregation Events</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <div className="overflow-hidden rounded-b-lg">
                            <table className="w-full text-sm text-left">
                                <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase text-xs">
                                    <tr>
                                        <th className="px-4 py-3 font-semibold">Version</th>
                                        <th className="px-4 py-3 font-semibold">Details</th>
                                        <th className="px-4 py-3 font-semibold text-right">Time</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {activityFeed.length > 0 ? activityFeed.map((row, i) => (
                                        <tr key={i} className="hover:bg-slate-50 transition-colors">
                                            <td className="px-4 py-3">
                                                <span className="px-2 py-0.5 rounded text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                                                    {row.details.split(" ")[0]}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 text-slate-500 text-xs">{row.details.split(" — ")[1] ?? row.details}</td>
                                            <td className="px-4 py-3 text-right text-slate-400 text-xs tabular-nums">{row.time}</td>
                                        </tr>
                                    )) : (
                                        <tr><td colSpan={3} className="px-4 py-8 text-center text-slate-400">No events yet.</td></tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
