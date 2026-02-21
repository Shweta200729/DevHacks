"use client";
import React, { useEffect, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BorderBeam } from "@/components/ui/border-beam";
import { Activity, Network, ShieldCheck, Cpu, ArrowUpRight, CopyCheck } from "lucide-react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend
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

    // Dataset Upload State
    const [uploadClientId, setUploadClientId] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [datasetUrl, setDatasetUrl] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<{ type: "success" | "error" | "info", msg: string } | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const fetchMetrics = async () => {
        try {
            const res = await fetch("http://localhost:8000/metrics");
            if (res.ok) {
                const json = await res.json();
                json.evaluations = json.evaluations.reverse();
                json.aggregations = json.aggregations.reverse();
                setData(json);
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

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleDatasetUpload = async () => {
        if (!uploadClientId || (!selectedFile && !datasetUrl)) {
            setUploadStatus({ type: "error", msg: "Please provide a Client ID and either a dataset file or URL." });
            return;
        }

        setIsUploading(true);
        setUploadStatus({ type: "info", msg: "Uploading and starting training..." });

        const formData = new FormData();
        formData.append("client_id", uploadClientId);
        if (selectedFile) formData.append("file", selectedFile);
        if (datasetUrl) formData.append("dataset_url", datasetUrl);

        try {
            const res = await fetch("http://localhost:8000/fl/api/dataset/upload", {
                method: "POST",
                body: formData,
            });

            if (res.ok) {
                setUploadStatus({ type: "success", msg: "Dataset uploaded! Background training started." });
                if (fileInputRef.current) fileInputRef.current.value = "";
                setSelectedFile(null);
                setDatasetUrl("");
            } else {
                const data = await res.json().catch(() => ({}));
                setUploadStatus({ type: "error", msg: data.detail || "Failed to upload dataset." });
            }
        } catch (e) {
            setUploadStatus({ type: "error", msg: "Network error occurred." });
        } finally {
            setIsUploading(false);
        }
    };

    const handleFetchWeights = async () => {
        if (!uploadClientId) {
            setUploadStatus({ type: "error", msg: "Please enter the Client ID to fetch weights." });
            return;
        }

        try {
            setUploadStatus({ type: "info", msg: "Checking weights..." });
            const res = await fetch(`http://localhost:8000/fl/api/model/weights/${uploadClientId}`);
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                setUploadStatus({
                    type: "error",
                    msg: data.detail || "Weights not found. Training may still be in progress."
                });
                return;
            }

            // If ok, trigger file download natively
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `weights_${uploadClientId}.pt`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            setUploadStatus({ type: "success", msg: "Weights downloaded successfully!" });
        } catch (e) {
            setUploadStatus({ type: "error", msg: "Network error occurred." });
        }
    };

    const calculateTotalClients = () => {
        if (!data?.aggregations) return { acc: 0, rej: 0, active: 0 };
        let acc = 0;
        let rej = 0;
        data.aggregations.forEach(a => {
            acc += a.total_accepted;
            rej += a.total_rejected; // If available directly, else calculate from DB logs
        });
        return { acc, rej, active: Math.max(1248, acc * 4) }; // Faux active scale for visual demo
    };

    const stats = calculateTotalClients();

    // Safety fallback latest values 
    const latestEval = data?.evaluations?.length ? data.evaluations[data.evaluations.length - 1] : null;
    const latestAgg = data?.aggregations?.length ? data.aggregations[data.aggregations.length - 1] : null;

    const kpis = [
        { title: "Current Model Version", value: `v${data?.current_version || 0}`, icon: Network, trend: "Live Synced" },
        { title: "Aggregation Method", value: latestAgg?.method || "Trimmed Mean", icon: Cpu, trend: "DP layer enabled" },
        { title: "Total Valid Updates", value: stats.acc.toString(), icon: Activity, trend: `Queue: ${data?.pending_queue_size || 0}` },
        { title: "Rejected Updates", value: stats.rej.toString(), icon: ShieldCheck, trend: "Malicious payloads blocked", highlight: true },
        { title: "Latest Accuracy", value: latestEval ? `${(latestEval.accuracy * 100).toFixed(2)}%` : "N/A", icon: CopyCheck, trend: "Validated on MNIST test set", highlightKey: true },
    ];

    // Build mixed activity timeline
    const activityFeed: any[] = [];
    if (data?.aggregations) {
        data.aggregations.slice().reverse().forEach((agg: any) => {
            activityFeed.push({
                event: "Global Model Updated",
                details: `Version v${agg.version_id} compiled using ${agg.method}`,
                status: "Success",
                time: `Round ${agg.version_id}`
            });
        });
    }

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex items-center justify-between">
                <div className="flex flex-col gap-2">
                    <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Global System Overview</h2>
                    <p className="text-slate-500">Real-time metrics for your federated learning infrastructure.</p>
                </div>
            </div>

            {/* KPI Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {kpis.map((kpi, i) => (
                    <Card key={i} className={`relative overflow-hidden transition-all duration-300 hover:shadow-md hover:-translate-y-0.5 ${kpi.highlightKey ? 'border-blue-200 bg-blue-50/30' : 'bg-white border-slate-200'}`}>
                        {kpi.highlightKey && <BorderBeam duration={8} size={150} />}
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium text-slate-500">{kpi.title}</CardTitle>
                            <kpi.icon className={`h-4 w-4 ${kpi.highlightKey ? "text-blue-600" : kpi.highlight ? "text-red-500" : "text-slate-400"}`} />
                        </CardHeader>
                        <CardContent>
                            <div className={`text-2xl font-bold ${kpi.highlightKey ? 'text-blue-700' : kpi.highlight ? 'text-red-600' : 'text-slate-900'}`}>{kpi.value}</div>
                            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                {i === 2 || i === 4 ? <ArrowUpRight className="h-3 w-3 text-green-500" /> : null}
                                {kpi.trend}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* User Real Local Dataset Upload Controller */}
            <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                <CardHeader>
                    <CardTitle className="text-xl font-bold text-slate-900 flex items-center gap-2">
                        <Cpu className="w-5 h-5 text-indigo-500" />
                        Local Dataset Training
                    </CardTitle>
                    <p className="text-slate-500 text-sm max-w-3xl mt-1">
                        Upload your personal dataset (.csv) or provide a remote URL. The server will spin up a decentralized training worker in the background exclusively containing your local data. Once training is complete, fetch your customized locally-trained weights (.pt).
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col gap-5">
                        <div className="flex flex-wrap gap-4 items-end">
                            <div className="flex flex-col gap-2">
                                <label className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Target Client ID</label>
                                <input
                                    className="bg-slate-50 border border-slate-200 text-slate-900 px-4 py-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-400 w-64 transition-all font-mono text-sm"
                                    value={uploadClientId}
                                    onChange={(e) => setUploadClientId(e.target.value)}
                                    placeholder="e.g. MyEdgeNode-1"
                                />
                            </div>

                            <div className="flex flex-col gap-2">
                                <label className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Dataset File OR URL</label>
                                <div className="flex flex-col gap-2">
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        onChange={handleFileChange}
                                        className="text-sm text-slate-500 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 transition-all cursor-pointer"
                                    />
                                    <div className="text-xs text-slate-400 font-medium tracking-wide">— OR —</div>
                                    <input
                                        type="text"
                                        placeholder="https://example.com/dataset.csv"
                                        value={datasetUrl}
                                        onChange={(e) => setDatasetUrl(e.target.value)}
                                        className="bg-slate-50 border border-slate-200 text-slate-900 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-400 w-64 transition-all text-sm"
                                    />
                                </div>
                            </div>

                            <div className="flex gap-3 ml-auto">
                                <button
                                    onClick={handleDatasetUpload}
                                    disabled={isUploading}
                                    className={`transition-all px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-sm ${isUploading
                                        ? 'bg-slate-200 text-slate-500 cursor-not-allowed'
                                        : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                                        }`}
                                >
                                    <ArrowUpRight className={`w-4 h-4 ${isUploading ? 'animate-bounce' : ''}`} />
                                    {isUploading ? 'Uploading...' : 'Upload & Train'}
                                </button>

                                <button
                                    onClick={handleFetchWeights}
                                    className="transition-all px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-sm border border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
                                >
                                    Fetch Weights
                                </button>
                            </div>
                        </div>

                        {uploadStatus && (
                            <div className={`p-4 rounded-lg text-sm font-medium border ${uploadStatus.type === "success" ? "bg-green-50 text-green-700 border-green-200" :
                                uploadStatus.type === "error" ? "bg-red-50 text-red-700 border-red-200" :
                                    "bg-blue-50 text-blue-700 border-blue-200"
                                }`}>
                                {uploadStatus.msg}
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Convergence (Accuracy vs Rounds)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data?.evaluations || []}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                    <XAxis dataKey="version_id" stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={(t) => `v${t}`} />
                                    <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(t) => `${(t * 100).toFixed(0)}%`} domain={[0, 1]} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        itemStyle={{ color: '#0f172a', fontWeight: 'bold' }}
                                        formatter={(value: any) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                                    />
                                    <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={3} dot={{ fill: "#2563eb", strokeWidth: 2, r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Global Training Loss</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50 p-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data?.evaluations || []}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                    <XAxis dataKey="version_id" stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={(t) => `v${t}`} />
                                    <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        itemStyle={{ color: '#0f172a', fontWeight: 'bold' }}
                                        formatter={(value: any) => [value.toFixed(4), 'Loss']}
                                    />
                                    <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={3} dot={{ fill: "#ef4444", strokeWidth: 2, r: 4 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Recent Activity Table */}
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
                                {activityFeed.length > 0 ? activityFeed.slice(0, 5).map((row, i) => (
                                    <tr key={i} className="bg-white hover:bg-slate-50 transition-colors">
                                        <td className="px-6 py-4 font-medium text-slate-900">{row.event}</td>
                                        <td className="px-6 py-4 text-slate-500">{row.details}</td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${row.status === 'Success' ? 'bg-green-50 text-green-700 border-green-200' :
                                                row.status === 'Warning' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                                                    'bg-blue-50 text-blue-700 border-blue-200'
                                                }`}>
                                                {row.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right text-slate-400 text-xs">{row.time}</td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan={4} className="px-6 py-8 text-center text-slate-500">No aggregation events yet. Fire up some simulated edge nodes!</td>
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
