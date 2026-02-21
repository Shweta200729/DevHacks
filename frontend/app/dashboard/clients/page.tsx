import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ShieldAlert, ShieldCheck, Activity, Users } from "lucide-react";

export default function ClientsPage() {
    const summary = [
        { title: "Total Clients", value: "2,048", icon: Users },
        { title: "Healthy Nodes", value: "1,980", icon: ShieldCheck, color: "text-green-600" },
        { title: "Suspicious", value: "44", icon: Activity, color: "text-amber-500" },
        { title: "Malicious Blocked", value: "24", icon: ShieldAlert, color: "text-red-500" },
    ];

    const clients = [
        { id: "node-cluster-a1", updates: 142, rejected: 0, trust: 0.99, status: "Healthy" },
        { id: "edge-device-88f", updates: 34, rejected: 12, trust: 0.45, status: "Suspicious" },
        { id: "server-pool-x9", updates: 890, rejected: 2, trust: 0.96, status: "Healthy" },
        { id: "unknown-origin-2", updates: 15, rejected: 15, trust: 0.12, status: "Malicious" },
        { id: "mobile-fleet-4a", updates: 512, rejected: 4, trust: 0.92, status: "Healthy" },
        { id: "iot-network-b2", updates: 120, rejected: 45, trust: 0.38, status: "Suspicious" },
        { id: "cloud-worker-99", updates: 2005, rejected: 0, trust: 1.00, status: "Healthy" },
        { id: "rogue-agent-x", updates: 50, rejected: 50, trust: 0.05, status: "Malicious" },
    ];

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Client Cohort Analysis</h2>
                <p className="text-slate-500">Monitor node health, activity, and trust scores across the federation.</p>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {summary.map((stat, i) => (
                    <Card key={i} className="bg-white border-slate-200 shadow-sm">
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium text-slate-500">{stat.title}</CardTitle>
                            <stat.icon className={`h-4 w-4 ${stat.color || "text-slate-400"}`} />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-slate-900">{stat.value}</div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Main Table */}
            <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                {/* Subtle decorative gradient */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-blue-50 rounded-full blur-3xl -z-0 translate-x-1/2 -translate-y-1/2" />

                <CardHeader className="relative z-10">
                    <CardTitle className="text-lg font-bold text-slate-900">Active Node Monitor</CardTitle>
                </CardHeader>
                <CardContent className="relative z-10">
                    <div className="w-full overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold uppercase text-xs tracking-wider">
                                <tr>
                                    <th className="px-6 py-4">Client ID</th>
                                    <th className="px-6 py-4 text-center">Total Updates</th>
                                    <th className="px-6 py-4 text-center">Rejected Updates</th>
                                    <th className="px-6 py-4">Trust Score (0-1)</th>
                                    <th className="px-6 py-4 text-right">Status</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {clients.map((client, i) => (
                                    <tr key={i} className="hover:bg-blue-50/30 transition-colors">
                                        <td className="px-6 py-4 font-mono text-xs text-blue-900 font-medium">
                                            {client.id}
                                        </td>
                                        <td className="px-6 py-4 text-center text-slate-700 font-medium">{client.updates.toLocaleString()}</td>
                                        <td className="px-6 py-4 text-center">
                                            <span className={`${client.rejected > 10 ? 'text-red-600 font-bold' : 'text-slate-600'}`}>
                                                {client.rejected.toLocaleString()}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-32 h-2 rounded-full bg-slate-100 overflow-hidden border border-slate-200">
                                                    <div
                                                        className={`h-full rounded-full ${client.trust > 0.8 ? 'bg-green-500' :
                                                                client.trust > 0.4 ? 'bg-amber-400' : 'bg-red-500'
                                                            }`}
                                                        style={{ width: `${client.trust * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-xs font-semibold text-slate-600">{client.trust.toFixed(2)}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${client.status === 'Healthy' ? 'bg-green-50 text-green-700 border-green-200 shadow-[0_0_10px_rgba(34,197,94,0.1)]' :
                                                    client.status === 'Suspicious' ? 'bg-amber-50 text-amber-700 border-amber-200 shadow-[0_0_10px_rgba(245,158,11,0.1)]' :
                                                        'bg-red-50 text-red-700 border-red-200 shadow-[0_0_10px_rgba(239,68,68,0.1)]'
                                                }`}>
                                                {client.status === 'Healthy' && <ShieldCheck className="w-3 h-3 mr-1" />}
                                                {client.status === 'Suspicious' && <Activity className="w-3 h-3 mr-1" />}
                                                {client.status === 'Malicious' && <ShieldAlert className="w-3 h-3 mr-1" />}
                                                {client.status}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
