<<<<<<< HEAD
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, GitMerge, Network } from "lucide-react";

export default function ModelsPage() {
    const versions = [
        { v: "v2.0.4", agg: "Trimmed Mean", clients: "1,024", acc: "94.2%", date: "Today, 14:32" },
        { v: "v2.0.3", agg: "Trimmed Mean", clients: "980", acc: "93.8%", date: "Today, 10:15" },
        { v: "v2.0.2", agg: "FedAvg", clients: "850", acc: "92.1%", date: "Yesterday, 18:40" },
        { v: "v2.0.1", agg: "FedAvg", clients: "845", acc: "91.5%", date: "Yesterday, 12:20" },
        { v: "v2.0.0", agg: "Trimmed Mean", clients: "1,200", acc: "89.0%", date: "Oct 24, 09:00", major: true },
        { v: "v1.9.5", agg: "FedAvg", clients: "750", acc: "88.5%", date: "Oct 20, 15:30" },
    ];
=======
"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, GitMerge, Network, CheckCircle2 } from "lucide-react";

interface ModelVersion {
    id: string;
    version_num: number;
    file_path: string;
    created_at: string;
}

interface EvalData {
    version_id: string;
    accuracy: number;
    loss: number;
}

interface AggData {
    version_id: string;
    method: string;
    total_accepted: number;
    total_rejected: number;
}

export default function ModelsPage() {
    const [versions, setVersions] = useState<ModelVersion[]>([]);
    const [evals, setEvals] = useState<Record<string, EvalData>>({});
    const [aggs, setAggs] = useState<Record<string, AggData>>({});
    const [downloading, setDownloading] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch("http://localhost:8000/fl/metrics");
                if (res.ok) {
                    const json = await res.json();
                    // Build lookup maps by version_id
                    const evalMap: Record<string, EvalData> = {};
                    json.evaluations?.forEach((e: EvalData) => { evalMap[e.version_id] = e; });
                    const aggMap: Record<string, AggData> = {};
                    json.aggregations?.forEach((a: AggData) => { aggMap[a.version_id] = a; });
                    setEvals(evalMap);
                    setAggs(aggMap);
                }

                const verRes = await fetch("http://localhost:8000/fl/versions");
                if (verRes.ok) {
                    const vJson = await verRes.json();
                    setVersions(vJson.data || []);
                }
            } catch (e) {
                console.error("Failed to fetch model data", e);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleDownload = async (versionId: string) => {
        setDownloading(versionId);
        try {
            const res = await fetch(`http://localhost:8000/fl/model/download?version_id=${versionId}`);
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `fl_global_model_v${versionId}.pt`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            }
        } catch (e) {
            alert("Download failed. Server may be unavailable.");
        } finally {
            setDownloading(null);
        }
    };
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Global Model Registry</h2>
<<<<<<< HEAD
                <p className="text-slate-500">Track and evaluate global model versions generated from asynchronous federation.</p>
            </div>

            <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                {/* Subtle decorative grid */}
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] mix-blend-overlay" />

                <CardHeader className="relative z-10">
                    <CardTitle className="text-lg font-bold text-slate-900">Version History</CardTitle>
                </CardHeader>
                <CardContent className="relative z-10">
=======
                <p className="text-slate-500">Track, evaluate, and download all global model versions generated from asynchronous federation.</p>
            </div>

            <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Version History</CardTitle>
                </CardHeader>
                <CardContent>
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
                    <div className="w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold uppercase text-xs tracking-wider">
                                <tr>
                                    <th className="px-6 py-4">Version</th>
                                    <th className="px-6 py-4">Aggregation Protocol</th>
<<<<<<< HEAD
                                    <th className="px-6 py-4 text-center">Clients Included</th>
                                    <th className="px-6 py-4 text-center">Validation Acc.</th>
                                    <th className="px-6 py-4 text-right">Compilation Date</th>
=======
                                    <th className="px-6 py-4 text-center">Valid / Rejected Clients</th>
                                    <th className="px-6 py-4 text-center">Validation Acc.</th>
                                    <th className="px-6 py-4 text-center">Loss</th>
                                    <th className="px-6 py-4 text-right">Compiled</th>
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
                                    <th className="px-6 py-4"></th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
<<<<<<< HEAD
                                {versions.map((version, i) => (
                                    <tr key={i} className={`transition-colors cursor-pointer group ${version.major ? 'bg-blue-50/50 hover:bg-blue-50' : 'hover:bg-slate-50'}`}>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2">
                                                {version.major ? <GitMerge className="w-4 h-4 text-blue-600" /> : <Network className="w-4 h-4 text-slate-400" />}
                                                <span className={`font-mono font-medium ${version.major ? 'text-blue-700 font-bold' : 'text-slate-900'}`}>
                                                    {version.v}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border ${version.agg === 'Trimmed Mean' ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-slate-100 text-slate-700 border-slate-200'
                                                }`}>
                                                {version.agg}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-center text-slate-700 font-medium">{version.clients}</td>
                                        <td className="px-6 py-4 text-center">
                                            <span className="inline-flex items-center justify-center bg-green-50 text-green-700 px-2 py-1 rounded text-xs font-bold border border-green-200">
                                                {version.acc}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right text-slate-500">{version.date}</td>
                                        <td className="px-6 py-4 text-right">
                                            <button className="text-slate-400 hover:text-blue-600 transition-colors opacity-0 group-hover:opacity-100">
                                                <Download className="w-5 h-5 inline-block" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
=======
                                {versions.length > 0 ? versions.map((version, i) => {
                                    const ev = evals[version.id];
                                    const ag = aggs[version.id];
                                    const isCurrent = i === 0;
                                    return (
                                        <tr key={version.id} className={`transition-colors cursor-pointer group ${isCurrent ? 'bg-blue-50/50 hover:bg-blue-50' : 'hover:bg-slate-50'}`}>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center gap-2">
                                                    {isCurrent ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : <Network className="w-4 h-4 text-slate-400" />}
                                                    <span className={`font-mono font-medium ${isCurrent ? 'text-blue-700 font-bold' : 'text-slate-900'}`}>
                                                        v{version.version_num}
                                                    </span>
                                                    {isCurrent && <span className="ml-1 px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-[10px] font-bold border border-green-200">LIVE</span>}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border ${ag?.method?.includes('Trimmed') ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-slate-100 text-slate-700 border-slate-200'}`}>
                                                    {ag?.method || 'N/A'}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-center text-slate-700 font-medium">
                                                {ag ? `${ag.total_accepted} / ${ag.total_rejected}` : '—'}
                                            </td>
                                            <td className="px-6 py-4 text-center">
                                                {ev ? (
                                                    <span className="inline-flex items-center justify-center bg-green-50 text-green-700 px-2 py-1 rounded text-xs font-bold border border-green-200">
                                                        {(ev.accuracy * 100).toFixed(2)}%
                                                    </span>
                                                ) : '—'}
                                            </td>
                                            <td className="px-6 py-4 text-center text-slate-500 font-mono text-xs">
                                                {ev ? ev.loss.toFixed(4) : '—'}
                                            </td>
                                            <td className="px-6 py-4 text-right text-slate-400 text-xs">
                                                {new Date(version.created_at).toLocaleString()}
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <button
                                                    onClick={() => handleDownload(version.id)}
                                                    disabled={downloading === version.id}
                                                    className="text-slate-400 hover:text-blue-600 transition-colors opacity-0 group-hover:opacity-100 disabled:opacity-50"
                                                    title="Download model weights (.pt)"
                                                >
                                                    <Download className={`w-5 h-5 inline-block ${downloading === version.id ? 'animate-bounce' : ''}`} />
                                                </button>
                                            </td>
                                        </tr>
                                    );
                                }) : (
                                    <tr>
                                        <td colSpan={7} className="px-6 py-8 text-center text-slate-500">
                                            No model versions found. Run a simulation to create the first global checkpoint!
                                        </td>
                                    </tr>
                                )}
>>>>>>> 9ea4d82af49c4f14145d1d31c2f41059b14ea187
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
