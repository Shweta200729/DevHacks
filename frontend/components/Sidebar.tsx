"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    LayoutDashboard,
    Users,
    Network,
    LineChart,
    TerminalSquare
} from "lucide-react";

export function Sidebar() {
    const pathname = usePathname();

    const links = [
        { name: "Overview", href: "/dashboard", icon: LayoutDashboard },
        { name: "Clients", href: "/dashboard/clients", icon: Users },
        { name: "Models", href: "/dashboard/models", icon: Network },
        { name: "Evaluation", href: "/dashboard/evaluation", icon: LineChart },
        { name: "Logs", href: "/dashboard/logs", icon: TerminalSquare },
    ];

    return (
        <div className="w-[240px] flex-shrink-0 bg-white border-r border-slate-200 min-h-screen flex flex-col pt-6 z-20 shadow-sm relative">
            <div className="px-6 mb-8 flex items-center gap-2">
                <div className="bg-blue-100 p-1.5 rounded-lg border border-blue-200">
                    <Network className="w-5 h-5 text-blue-600" />
                </div>
                <span className="font-bold text-lg text-slate-900 tracking-tight">AsyncFL</span>
            </div>

            <nav className="flex flex-col gap-1 px-3">
                {links.map((link) => {
                    // Exact match for overview, startsWith for others to keep active state on subpages
                    const isActive = link.href === "/dashboard"
                        ? pathname === "/dashboard"
                        : pathname.startsWith(link.href);

                    return (
                        <Link
                            key={link.name}
                            href={link.href}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-xl font-medium transition-all group relative overflow-hidden ${isActive
                                    ? "text-blue-700 bg-blue-50/80 shadow-sm border border-blue-100"
                                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent"
                                }`}
                        >
                            {isActive && (
                                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-blue-600 rounded-r-full" />
                            )}
                            <link.icon className={`w-5 h-5 ${isActive ? "text-blue-600" : "text-slate-400 group-hover:text-slate-600"}`} />
                            {link.name}
                        </Link>
                    );
                })}
            </nav>

            <div className="mt-auto p-4 m-4 rounded-xl bg-slate-50 border border-slate-200 relative overflow-hidden">
                <div className="absolute inset-0 bg-blue-500/5 blur-2xl rounded-full" />
                <p className="text-xs font-semibold text-slate-900 mb-1 relative z-10">Workspace Status</p>
                <div className="flex items-center gap-2 text-xs text-slate-500 relative z-10">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    System Healthy
                </div>
            </div>
        </div>
    );
}
