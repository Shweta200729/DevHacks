"use client";
import React, { useState, useEffect } from "react";
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
            const res = await fetch("http://localhost:8000/clients");
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
            await fetch("http://localhost:8000/simulate", {
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
