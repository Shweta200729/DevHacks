"use client";
import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PlayCircle, ShieldAlert, Cpu, ArrowUpRight, CopyCheck, AlertTriangle } from "lucide-react";

interface ClientUpdate {
    client_id: string;
    status: string;
    norm_value: number | null;
    distance_value: number | null;
    reason: string;
    created_at?: string;
}

export default function ClientsPage() {
    const [simName, setSimName] = useState("");
    const [isMalicious, setIsMalicious] = useState(false);
    const [isSimulating, setIsSimulating] = useState(false);
    const [clientUpdates, setClientUpdates] = useState<ClientUpdate[]>([]);

    // Dataset Upload State
    const [uploadClientId, setUploadClientId] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [datasetUrl, setDatasetUrl] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<{ type: "success" | "error" | "info", msg: string } | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    // Auto-generate realistic client ID on load
    useEffect(() => {
        setSimName(`EdgeNode-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`);
        fetchClientUpdates();
        const interval = setInterval(fetchClientUpdates, 3000);
        return () => clearInterval(interval);
    }, []);

    const fetchClientUpdates = async () => {
        try {
            // Need a new backend route or we can derive from existing. 
            // For now, let's fetch /metrics and if we add a dedicated endpoint later, we can swap.
            // Actually, we need to create a dedicated /clients endpoint in main.py for this table.
            // For demo purposes, we will fetch standard metrics to ensure the system is alive
            // and fallback to a mock array if the endpoint isn't built yet, but we WILL build it next.
            const res = await fetch("http://localhost:8000/fl/clients");
            if (res.ok) {
                const json = await res.json();
                setClientUpdates(json.data || []);
            }
        } catch (e) {
            console.error("Failed fetching clients", e);
        }
    };

    const handleSimulate = async () => {
        setIsSimulating(true);
        try {
            await fetch("http://localhost:8000/fl/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    client_name: simName,
                    is_malicious: isMalicious,
                    malicious_multiplier: 50.0,
                }),
            });
            // Update name for next quick fire
            setSimName(`EdgeNode-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`);
            // Optimistic fetch delay
            setTimeout(fetchClientUpdates, 1500);
        } catch (e) {
            alert("Simulation request failed to reach server.");
        } finally {
            setIsSimulating(false);
        }
    };

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
                setUploadStatus({ type: "info", msg: "Dataset submitted. Training model on isolated local data..." });
                if (fileInputRef.current) fileInputRef.current.value = "";
                setSelectedFile(null);
                setDatasetUrl("");

                // Poll the weights endpoint until the background task completes
                const currentClientId = uploadClientId;
                const pollInterval = setInterval(async () => {
                    try {
                        const checkRes = await fetch(`http://localhost:8000/fl/api/model/weights/${currentClientId}`);
                        if (checkRes.ok) {
                            clearInterval(pollInterval);
                            setUploadStatus({ type: "success", msg: "Local training complete! Download initiated." });

                            // Automatically download weights natively
                            const url = `http://localhost:8000/fl/api/model/weights/${currentClientId}`;
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `weights_${currentClientId}.pt`;
                            document.body.appendChild(a);
                            a.click();
                            a.remove();

                            setIsUploading(false);
                        }
                    } catch (e) {
                        // Suppress polling connection drops
                    }
                }, 2000);

                // Timeout switch in case training stalls heavily (> 60s)
                setTimeout(() => {
                    clearInterval(pollInterval);
                    setIsUploading(false);
                }, 60000);

            } else {
                const data = await res.json();
                setUploadStatus({ type: "error", msg: data.detail || "Failed to upload dataset." });
                setIsUploading(false);
            }
        } catch (e) {
            setUploadStatus({ type: "error", msg: "Network error occurred." });
            setIsUploading(false);
        }
    };

    const handleFetchWeights = async () => {
        if (!uploadClientId) {
            setUploadStatus({ type: "error", msg: "Please enter the Client ID to fetch weights." });
            return;
        }
        setUploadStatus({ type: "info", msg: "Checking weights..." });

        try {
            // First do a lightweight check to ensure the file exists and get the backend error if it doesn't
            const res = await fetch(`http://localhost:8000/fl/api/model/weights/${uploadClientId}`);
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                setUploadStatus({
                    type: "error",
                    msg: data.detail || "Weights not found. Training may still be in progress."
                });
                return;
            }

            // If ok, use native browser fetching to bypass fetch stream issues
            const url = `http://localhost:8000/fl/api/model/weights/${uploadClientId}`;
            const a = document.createElement("a");
            a.href = url;
            a.download = `weights_${uploadClientId}.pt`;
            document.body.appendChild(a);
            a.click();
            a.remove();

            setTimeout(() => setUploadStatus({ type: "success", msg: "Weights downloaded successfully!" }), 1000);
        } catch (e) {
            setUploadStatus({ type: "error", msg: "Network error occurred." });
        }
    };

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Edge Clients & Tokens</h2>
                <p className="text-slate-500">Manage connected nodes, simulate updates, and monitor SLT Token slashes.</p>
            </div>

            {/* Admin Simulation Controller */}
            <Card className="bg-indigo-950 border-indigo-900 shadow-xl shadow-indigo-900/10 overflow-hidden relative">
                {/* Decorative background glow */}
                <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/20 blur-3xl rounded-full pointer-events-none" />

                <CardHeader>
                    <CardTitle className="text-xl font-bold text-indigo-100 flex items-center gap-2">
                        <ShieldAlert className="w-5 h-5 text-indigo-400" />
                        Simulation Controller
                    </CardTitle>
                    <p className="text-indigo-200/70 text-sm max-w-2xl mt-1">
                        Inject authentic edge client training updates directly into the FastAPI endpoint.
                        Normal clients train a CNN randomly over an isolated MNIST batch.
                        Malicious clients apply harsh noise logic generating high L2 bounds to test the Byzantine defenses.
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-wrap gap-4 items-end relative z-10">
                        <div className="flex flex-col gap-2">
                            <label className="text-xs text-indigo-300 font-semibold uppercase tracking-wider">Client Node ID</label>
                            <input
                                className="bg-indigo-900/50 border border-indigo-700/50 text-white px-4 py-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-400 w-64 placeholder:text-indigo-700 transition-all font-mono text-sm"
                                value={simName}
                                onChange={(e) => setSimName(e.target.value)}
                                placeholder="e.g. EdgeNode-139"
                            />
                        </div>

                        <div className="flex items-center gap-3 bg-indigo-900/30 border border-indigo-800/50 px-5 py-2.5 rounded-lg h-[42px]">
                            <input
                                type="checkbox"
                                id="maliciousToggle"
                                checked={isMalicious}
                                onChange={(e) => setIsMalicious(e.target.checked)}
                                className="w-4 h-4 rounded border-indigo-700 text-red-500 focus:ring-red-500 bg-indigo-950 cursor-pointer accent-red-500"
                            />
                            <label htmlFor="maliciousToggle" className={`text-sm font-medium cursor-pointer transition-colors ${isMalicious ? 'text-red-400' : 'text-indigo-300 hover:text-indigo-200'}`}>
                                Inject Byzantine Data Poisoning
                            </label>
                        </div>

                        <button
                            onClick={handleSimulate}
                            disabled={isSimulating}
                            className={`transition-all px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-lg h-[42px] ${isMalicious
                                ? 'bg-red-600 hover:bg-red-700 text-white shadow-red-900/50'
                                : 'bg-indigo-500 hover:bg-indigo-600 text-white shadow-indigo-900/50'
                                } ${isSimulating ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            <PlayCircle className={`w-5 h-5 ${isSimulating ? 'animate-spin' : ''}`} />
                            {isSimulating ? 'Simulating...' : isMalicious ? 'Fire Malicious Payload' : 'Fire Normal Update'}
                        </button>
                    </div>
                </CardContent>
            </Card>

            {/* Dataset Upload & Weights Download */}
            <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                <CardHeader>
                    <CardTitle className="text-xl font-bold text-slate-900 flex items-center gap-2">
                        <Cpu className="w-5 h-5 text-blue-500" />
                        Client Decentralized Training
                    </CardTitle>
                    <p className="text-slate-500 text-sm max-w-3xl mt-1">
                        Upload your personal structured dataset securely. The server will spin up a decentralized training worker in the background exclusively containing your local data. Once training is complete, fetch your isolated locally-trained weights (.pt).
                    </p>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col gap-6">
                        <div className="flex flex-col md:flex-row gap-4">
                            <div className="flex flex-col gap-2 flex-1">
                                <label className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Target Client ID</label>
                                <input
                                    className="bg-slate-50 border border-slate-200 text-slate-900 px-4 py-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 w-full placeholder:text-slate-400 transition-all text-sm"
                                    value={uploadClientId}
                                    onChange={(e) => setUploadClientId(e.target.value)}
                                    placeholder="e.g. MyEdgeNode-1"
                                />
                            </div>
                            <div className="flex flex-col gap-2 flex-1 relative">
                                <label className="text-xs text-slate-500 font-semibold uppercase tracking-wider">CSV Dataset URL (Optional)</label>
                                <input
                                    className="bg-slate-50 border border-slate-200 text-slate-900 px-4 py-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 w-full placeholder:text-slate-400 transition-all text-sm"
                                    value={datasetUrl}
                                    onChange={(e) => setDatasetUrl(e.target.value)}
                                    placeholder="https://example.com/dataset.csv"
                                />
                            </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-4">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                className="hidden"
                                ref={fileInputRef}
                            />
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="bg-slate-100 hover:bg-slate-200 text-slate-700 px-5 py-2.5 rounded-lg font-medium transition-colors border border-slate-200 text-sm shadow-sm"
                            >
                                {selectedFile ? selectedFile.name : "Select Local CSV"}
                            </button>

                            <button
                                onClick={handleDatasetUpload}
                                disabled={isUploading}
                                className={`transition-all px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-sm text-sm ${isUploading ? 'bg-slate-300 text-slate-500 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-600/20'
                                    }`}
                            >
                                {isUploading ? "Uploading..." : "Upload & Train"}
                            </button>

                            <div className="h-8 w-px bg-slate-200 mx-2" />

                            <button
                                onClick={handleFetchWeights}
                                className="transition-all px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-sm text-sm bg-slate-800 hover:bg-slate-900 text-white shadow-slate-800/20"
                            >
                                Fetch Weights
                            </button>
                        </div>

                        {uploadStatus && (
                            <div className={`text-sm font-medium px-4 py-3 rounded-lg flex items-center gap-2 ${uploadStatus.type === "error" ? "bg-red-50 text-red-700 border border-red-200" :
                                uploadStatus.type === "success" ? "bg-green-50 text-green-700 border border-green-200" :
                                    "bg-blue-50 text-blue-700 border border-blue-200"
                                }`}>
                                {uploadStatus.type === "error" && <AlertTriangle className="w-4 h-4" />}
                                {uploadStatus.type === "success" && <CopyCheck className="w-4 h-4" />}
                                {uploadStatus.type === "info" && <Cpu className="w-4 h-4 animate-pulse" />}
                                {uploadStatus.msg}
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Clients Table */}
            <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader>
                    <CardTitle className="text-lg font-bold text-slate-900">Live Client Update Stream</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-50/50">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-100/50 text-slate-500 uppercase text-xs">
                                <tr>
                                    <th className="px-6 py-3 font-semibold">Client ID</th>
                                    <th className="px-6 py-3 font-semibold">Defense Status</th>
                                    <th className="px-6 py-3 font-semibold">L2 Norm / Distance</th>
                                    <th className="px-6 py-3 font-semibold text-right">Blockchain Incentive</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200">
                                {clientUpdates.length > 0 ? clientUpdates.map((row, i) => (
                                    <tr key={i} className="bg-white hover:bg-slate-50 transition-colors">
                                        <td className="px-6 py-4 font-mono text-slate-700 text-xs font-semibold">{row.client_id}</td>
                                        <td className="px-6 py-4">
                                            {row.status === "ACCEPT" ? (
                                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">
                                                    <CopyCheck className="w-3.5 h-3.5" />
                                                    Validated
                                                </span>
                                            ) : (
                                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200" title={row.reason}>
                                                    <AlertTriangle className="w-3.5 h-3.5" />
                                                    Rejected
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-slate-500 font-mono text-xs">
                                            {row.norm_value ? (row.norm_value).toFixed(2) : '-'} / {row.distance_value ? (row.distance_value).toFixed(2) : '-'}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            {row.status === "ACCEPT" ? (
                                                <span className="font-bold text-green-600">+10 FLT</span>
                                            ) : (
                                                <span className="font-bold text-red-600">-15 FLT (Slashed)</span>
                                            )}
                                        </td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan={4} className="px-6 py-8 text-center text-slate-500">
                                            No client runs detected in the current active global round. Simulate an update above!
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
