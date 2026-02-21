import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BorderBeam } from "@/components/ui/border-beam";
import { Activity, Network, ShieldCheck, Cpu, ArrowUpRight, CopyCheck } from "lucide-react";

export default function OverviewPage() {
    const kpis = [
        { title: "Current Model Version", value: "v2.0.4", icon: Network, trend: "+1 new since yesterday" },
        { title: "Aggregation Method", value: "Trimmed Mean", icon: Cpu, trend: "Robust enabled" },
        { title: "Active Clients", value: "1,248", icon: Activity, trend: "+12% active this hour" },
        { title: "Rejected Updates", value: "24", icon: ShieldCheck, trend: "0.02% rejection rate", highlight: true },
        { title: "Latest Accuracy", value: "94.2%", icon: CopyCheck, trend: "+0.4% from last global round", highlightKey: true },
    ];

    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Global System Overview</h2>
                <p className="text-slate-500">Real-time metrics for your federated learning infrastructure.</p>
            </div>

            {/* KPI Section */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {kpis.map((kpi, i) => (
                    <Card key={i} className={`relative overflow-hidden transition-all duration-300 hover:shadow-md hover:-translate-y-0.5 ${kpi.highlightKey ? 'border-blue-200 bg-blue-50/30' : 'bg-white border-slate-200'}`}>
                        {kpi.highlightKey && <BorderBeam duration={8} size={150} />}
                        <CardHeader className="flex flex-row items-center justify-between pb-2">
                            <CardTitle className="text-sm font-medium text-slate-500">{kpi.title}</CardTitle>
                            <kpi.icon className={`h-4 w-4 ${kpi.highlightKey ? "text-blue-600" : "text-slate-400"}`} />
                        </CardHeader>
                        <CardContent>
                            <div className={`text-2xl font-bold ${kpi.highlightKey ? 'text-blue-700' : 'text-slate-900'}`}>{kpi.value}</div>
                            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                                {i === 2 || i === 4 ? <ArrowUpRight className="h-3 w-3 text-green-500" /> : null}
                                {kpi.trend}
                            </p>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Convergence (Accuracy vs Rounds)</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50">
                            {/* Simulated Chart SVG */}
                            <svg className="absolute bottom-0 w-full h-full drop-shadow-[0_0_10px_rgba(37,99,235,0.1)]" viewBox="0 0 1000 300" preserveAspectRatio="none">
                                <path
                                    d="M 50 280 Q 150 250 250 150 T 450 100 T 650 60 T 850 40 T 950 35 L 950 300 L 50 300 Z"
                                    fill="url(#accGradient)"
                                    className="opacity-60"
                                />
                                <path
                                    d="M 50 280 Q 150 250 250 150 T 450 100 T 650 60 T 850 40 T 950 35"
                                    fill="none"
                                    stroke="#2563EB"
                                    strokeWidth="3"
                                />
                                <defs>
                                    <linearGradient id="accGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#2563EB" stopOpacity="0.2" />
                                        <stop offset="100%" stopColor="#2563EB" stopOpacity="0" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            {/* Axis markers */}
                            <div className="absolute inset-0 border-b border-l border-slate-200 m-4 pointer-events-none" />
                        </div>
                    </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                    <CardHeader>
                        <CardTitle className="text-lg font-bold text-slate-900">Global Training Loss</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-64 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50">
                            {/* Simulated Chart SVG */}
                            <svg className="absolute bottom-0 w-full h-full drop-shadow-[0_0_10px_rgba(37,99,235,0.1)]" viewBox="0 0 1000 300" preserveAspectRatio="none">
                                <path
                                    d="M 50 50 Q 150 80 250 150 T 450 220 T 650 250 T 850 270 T 950 280 L 950 300 L 50 300 Z"
                                    fill="url(#lossGradient)"
                                    className="opacity-60"
                                />
                                <path
                                    d="M 50 50 Q 150 80 250 150 T 450 220 T 650 250 T 850 270 T 950 280"
                                    fill="none"
                                    stroke="#3B82F6"
                                    strokeWidth="3"
                                />
                                <defs>
                                    <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.15" />
                                        <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            {/* Axis markers */}
                            <div className="absolute inset-0 border-b border-l border-slate-200 m-4 pointer-events-none" />
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
                                {[
                                    { event: "Global Model Updated", details: "Version v2.0.4 compiled using Trimmed Mean", status: "Success", time: "2 mins ago" },
                                    { event: "Update Rejected", details: "Client #894A L2-norm exceeded threshold (15.4 > 10.0)", status: "Warning", time: "12 mins ago" },
                                    { event: "Aggregation Started", details: "Waiting for 100 client updates", status: "Info", time: "45 mins ago" },
                                    { event: "Client Registered", details: "New edge node cluster connected in US-West", status: "Success", time: "1 hour ago" },
                                ].map((row, i) => (
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
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
