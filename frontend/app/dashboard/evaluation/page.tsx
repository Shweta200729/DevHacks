"use client";
import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend,
    AreaChart, Area,
} from "recharts";
import { fetchMetrics, EvalRow, startPolling } from "@/lib/api";

interface ChartRow {
    version_id: number;
    accuracy: number;
    loss: number;
}

export default function EvaluationPage() {
    const [rows, setRows] = useState<ChartRow[]>([]);
    const [loading, setLoading] = useState(true);

    const load = useCallback(async () => {
        const json = await fetchMetrics();
        if (json) {
            // oldest → newest for left-to-right charts
            setRows(
                [...json.evaluations]
                    .reverse()
                    .map(e => ({ version_id: e.version_id, accuracy: e.accuracy, loss: e.loss }))
            );
        }
        setLoading(false);
    }, []);

    useEffect(() => startPolling(load, 3000), [load]);

    const latest = rows.at(-1);
    const best = rows.length > 0
        ? rows.reduce((best, r) => r.accuracy > best.accuracy ? r : best)
        : null;

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Model Evaluation</h2>
                <p className="text-slate-500">
                    Real accuracy and loss curves tracked after every federated aggregation round.
                </p>
            </div>

            {/* Stat pills */}
            {rows.length > 0 && (
                <div className="flex flex-wrap gap-4">
                    {[
                        { label: "Rounds completed", value: rows.length.toString() },
                        { label: "Latest accuracy", value: `${(latest!.accuracy * 100).toFixed(2)}%` },
                        { label: "Latest loss", value: latest!.loss.toFixed(4) },
                        { label: "Best accuracy (round)", value: `${(best!.accuracy * 100).toFixed(2)}% @ v${best!.version_id}` },
                    ].map((s, i) => (
                        <div key={i} className="bg-white border border-slate-200 rounded-xl px-5 py-3 flex flex-col gap-0.5 shadow-sm">
                            <span className="text-xs text-slate-500 font-medium uppercase tracking-wide">{s.label}</span>
                            <span className="text-lg font-bold text-slate-900 font-mono">{s.value}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Accuracy Chart */}
            <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Accuracy per Round</CardTitle>
                    <p className="text-sm text-slate-500">
                        Real MNIST test-set accuracy evaluated after each FedAvg / DP-Trimmed Mean aggregation.
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="h-[320px] w-full rounded-xl border border-slate-100 bg-slate-50/50 p-4">
                        {rows.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={rows}>
                                    <defs>
                                        <linearGradient id="accGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                                    <XAxis dataKey="version_id" stroke="#64748b" tickFormatter={t => `v${t}`} />
                                    <YAxis stroke="#64748b" domain={[0, 1]} tickFormatter={t => `${(t * 100).toFixed(0)}%`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", boxShadow: "0 4px 6px -1px rgba(0,0,0,.1)" }}
                                        formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, "Accuracy"]}
                                        labelFormatter={l => `Round v${l}`}
                                    />
                                    <Area type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={3}
                                        fill="url(#accGrad)" dot={{ fill: "#3b82f6", r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex h-full items-center justify-center text-slate-400 text-sm">
                                {loading ? "Connecting to server…" : "No evaluation data yet. Run a simulation to start."}
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Loss Chart */}
            <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Cross-Entropy Loss per Round</CardTitle>
                    <p className="text-sm text-slate-500">
                        Lower is better. Measures how well the global model fits the MNIST validation distribution.
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="h-[320px] w-full rounded-xl border border-slate-100 bg-slate-50/50 p-4">
                        {rows.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={rows}>
                                    <defs>
                                        <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.15} />
                                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                                    <XAxis dataKey="version_id" stroke="#64748b" tickFormatter={t => `v${t}`} />
                                    <YAxis stroke="#64748b" />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", boxShadow: "0 4px 6px -1px rgba(0,0,0,.1)" }}
                                        formatter={(v: number) => [v.toFixed(4), "Loss"]}
                                        labelFormatter={l => `Round v${l}`}
                                    />
                                    <Area type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={3}
                                        fill="url(#lossGrad)" dot={{ fill: "#ef4444", r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex h-full items-center justify-center text-slate-400 text-sm">
                                {loading ? "Connecting to server…" : "No evaluation data yet."}
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Raw data table */}
            {rows.length > 0 && (
                <Card className="bg-white border-slate-200 shadow-sm">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Evaluation History</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-hidden rounded-lg border border-slate-200">
                            <table className="w-full text-sm text-left">
                                <thead className="bg-slate-50 border-b border-slate-200 text-xs text-slate-500 uppercase font-semibold tracking-wider">
                                    <tr>
                                        <th className="px-6 py-3">Round</th>
                                        <th className="px-6 py-3 text-center">Accuracy</th>
                                        <th className="px-6 py-3 text-center">Loss</th>
                                        <th className="px-6 py-3 text-right">Δ Accuracy</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {[...rows].reverse().map((r, i, arr) => {
                                        const prev = arr[i + 1];
                                        const delta = prev ? r.accuracy - prev.accuracy : null;
                                        return (
                                            <tr key={r.version_id} className={`hover:bg-slate-50 transition-colors ${i === 0 ? "bg-blue-50/30" : ""}`}>
                                                <td className="px-6 py-3 font-mono font-semibold text-slate-700">
                                                    v{r.version_id}
                                                    {i === 0 && <span className="ml-2 text-[10px] px-1.5 py-0.5 bg-green-100 text-green-700 rounded-full font-bold">LATEST</span>}
                                                </td>
                                                <td className="px-6 py-3 text-center">
                                                    <span className="font-bold text-blue-600">{(r.accuracy * 100).toFixed(2)}%</span>
                                                </td>
                                                <td className="px-6 py-3 text-center font-mono text-slate-500">{r.loss.toFixed(4)}</td>
                                                <td className="px-6 py-3 text-right font-mono font-semibold">
                                                    {delta !== null ? (
                                                        <span className={delta >= 0 ? "text-green-600" : "text-red-500"}>
                                                            {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(2)}%
                                                        </span>
                                                    ) : <span className="text-slate-400">—</span>}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
