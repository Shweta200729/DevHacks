import { GitMerge, LineChart, ShieldCheck, Lock } from 'lucide-react';

const features = [
    {
        icon: <GitMerge className="w-7 h-7 text-blue-600" />,
        title: "Async Client Updates",
        description: "Edge nodes push gradient updates independently. The master node queues updates non-blockingly — no slow client can stall the global round.",
        tag: "Implemented"
    },
    {
        icon: <ShieldCheck className="w-7 h-7 text-blue-600" />,
        title: "Byzantine Detection",
        description: "Two-gate L2 defence: updates with a suspiciously high norm or drift from the global model are detected and rejected before entering the aggregation queue.",
        tag: "Implemented"
    },
    {
        icon: <LineChart className="w-7 h-7 text-blue-600" />,
        title: "Trimmed Mean Aggregation",
        description: "After detection, the remaining valid updates are aggregated using Trimmed Mean — a Byzantine-robust algorithm that discards the top & bottom 20% of values.",
        tag: "Implemented"
    },
    {
        icon: <Lock className="w-7 h-7 text-blue-600" />,
        title: "Differential Privacy",
        description: "Gaussian noise is injected post-aggregation using the DP-SGD formulation (σ = noise_mult × clip_norm). This protects individual client data mathematically.",
        tag: "Implemented"
    }
];

export default function Features() {
    return (
        <section id="features" className="py-24 bg-transparent">
            <div className="container mx-auto px-4 relative z-10">

                <div className="text-center mb-16">
                    <span className="inline-block px-4 py-1.5 rounded-full bg-blue-100 text-blue-700 text-xs font-bold tracking-widest uppercase mb-4 border border-blue-200">Core Architecture</span>
                    <h2 className="text-3xl md:text-5xl font-extrabold mb-4 text-slate-900 tracking-tight">Production-Grade Federated Learning</h2>
                    <p className="text-slate-600 text-lg max-w-2xl mx-auto leading-relaxed">A complete asynchronous FL pipeline — from edge training to secure, privacy-preserving global aggregation. Built for reproducible research and real deployment.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className="bg-linear-to-br from-white to-blue-50/30 border border-slate-200 border-t-4 border-t-blue-400 p-8 rounded-2xl shadow-sm hover:-translate-y-2 hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300 group relative overflow-hidden"
                        >
                            <div className="absolute top-3 right-3">
                                <span className="text-[10px] font-bold text-green-600 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full">{feature.tag}</span>
                            </div>
                            <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                {feature.icon}
                            </div>
                            <h3 className="text-xl font-bold mb-3 text-slate-900">{feature.title}</h3>
                            <p className="text-slate-600 leading-relaxed text-sm">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </div>

            </div>
        </section>
    );
}
