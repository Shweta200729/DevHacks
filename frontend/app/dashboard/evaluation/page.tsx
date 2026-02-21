import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function EvaluationPage() {
    return (
        <div className="flex flex-col gap-8 pb-10">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Aggregation Evaluation</h2>
                <p className="text-slate-500">Compare robust aggregation methods against baseline performance metrics.</p>
            </div>

            <Tabs defaultValue="robust" className="w-full">
                <div className="flex items-center justify-between mb-6">
                    <TabsList className="bg-slate-200/50 p-1 border border-slate-200">
                        <TabsTrigger value="baseline" className="rounded-md data-[state=active]:bg-white data-[state=active]:text-slate-900 data-[state=active]:shadow-sm">Baseline (FedAvg)</TabsTrigger>
                        <TabsTrigger value="robust" className="rounded-md data-[state=active]:bg-white data-[state=active]:text-blue-700 data-[state=active]:shadow-sm border border-transparent data-[state=active]:border-blue-100">Robust (Trimmed Mean)</TabsTrigger>
                    </TabsList>
                </div>

                {/* Robust Evaluation Content */}
                <TabsContent value="robust" className="space-y-6 mt-0 border-none outline-none">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                        {/* Accuracy Comparison Chart */}
                        <Card className="bg-white border-blue-100 shadow-md relative overflow-hidden ring-1 ring-blue-50">
                            <CardHeader>
                                <CardTitle className="text-lg font-bold text-slate-900">Convergence Stability</CardTitle>
                                <CardDescription>Accuracy trajectory highlighting resilience to Byzantine nodes.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="h-72 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50">
                                    <svg className="absolute bottom-0 w-full h-full drop-shadow-[0_0_10px_rgba(37,99,235,0.1)]" viewBox="0 0 1000 300" preserveAspectRatio="none">
                                        {/* Baseline path (background) */}
                                        <path
                                            d="M 50 280 L 150 200 L 250 220 L 350 150 L 450 180 L 550 120 L 650 140 L 750 90 L 850 100 L 950 80"
                                            fill="none"
                                            stroke="#94A3B8"
                                            strokeWidth="2"
                                            strokeDasharray="4 4"
                                        />
                                        {/* Robust path (foreground) */}
                                        <path
                                            d="M 50 280 Q 150 200 250 130 T 450 80 T 650 50 T 850 35 T 950 30"
                                            fill="none"
                                            stroke="#2563EB"
                                            strokeWidth="3"
                                        />
                                        {/* Fill */}
                                        <path
                                            d="M 50 280 Q 150 200 250 130 T 450 80 T 650 50 T 850 35 T 950 30 L 950 300 L 50 300 Z"
                                            fill="url(#accGradientRob)"
                                            className="opacity-40"
                                        />
                                        <defs>
                                            <linearGradient id="accGradientRob" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#2563EB" stopOpacity="0.3" />
                                                <stop offset="100%" stopColor="#2563EB" stopOpacity="0" />
                                            </linearGradient>
                                        </defs>
                                    </svg>
                                    <div className="absolute top-4 right-4 flex flex-col gap-2 text-xs">
                                        <div className="flex items-center gap-2"><div className="w-3 h-0.5 bg-blue-600"></div> Trimmed Mean</div>
                                        <div className="flex items-center gap-2"><div className="w-3 h-0.5 bg-slate-400 border-dashed border"></div> FedAvg</div>
                                    </div>
                                    <div className="absolute inset-0 border-b border-l border-slate-200 m-4 pointer-events-none" />
                                </div>
                            </CardContent>
                        </Card>

                        {/* Attack Resilience Chart */}
                        <Card className="bg-white border-slate-200 shadow-sm relative overflow-hidden">
                            <CardHeader>
                                <CardTitle className="text-lg font-bold text-slate-900">Attack Resilience Scoring</CardTitle>
                                <CardDescription>Impact of malicious updates on global model weights.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="h-72 w-full relative flex items-end pt-4 rounded-xl border border-slate-100 bg-slate-50/50 p-4">
                                    {/* Simulated bar chart */}
                                    <div className="w-full h-full flex items-end justify-around gap-2 px-4 pb-4">
                                        {[12, 18, 15, 8, 45, 12, 10, 8].map((val, idx) => (
                                            <div key={idx} className="relative group w-12 flex flex-col justify-end items-center h-full">
                                                {/* Shadow bar for baseline */}
                                                <div className="absolute bottom-0 w-8 bg-slate-200 rounded-t-sm" style={{ height: `${val * 1.5}%` }}></div>
                                                {/* Primary bar for robust */}
                                                <div className="absolute bottom-0 w-8 bg-blue-500 rounded-t-sm shadow-md" style={{ height: `${val}%` }}></div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* Baseline Evaluation Content (Placeholder for demo) */}
                <TabsContent value="baseline" className="space-y-6 mt-0 border-none outline-none">
                    <Card className="bg-slate-50 border-slate-200 border-dashed shadow-none p-12 text-center flex flex-col items-center justify-center">
                        <h3 className="text-xl font-bold text-slate-400">Baseline metrics disabled</h3>
                        <p className="text-slate-500 mt-2 max-w-sm">Switch to the Robust tab to view production metrics using the current Trimmed Mean configuration.</p>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    );
}
