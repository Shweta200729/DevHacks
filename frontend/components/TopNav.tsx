"use client";

import { Bell, Search, User } from "lucide-react";
import { usePathname } from "next/navigation";

export function TopNav() {
    const pathname = usePathname();

    // Create a pleasant title from the pathname
    const pathSegments = pathname.split("/").filter(Boolean);
    let title = "Overview";
    if (pathSegments.length > 1) {
        const lastSegment = pathSegments[pathSegments.length - 1];
        title = lastSegment.charAt(0).toUpperCase() + lastSegment.slice(1);
    }

    return (
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center justify-between px-8 sticky top-0 z-10 shadow-sm">
            <div className="flex items-center gap-4">
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">{title}</h1>
                <div className="h-5 w-px bg-slate-300 mx-2" />
                <span className="text-sm font-medium text-slate-500 bg-slate-100 px-2.5 py-1 rounded-full border border-slate-200">
                    Project: Alpha-Fold Variant
                </span>
            </div>

            <div className="flex items-center gap-4">
                <div className="relative hidden md:block">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search activity..."
                        className="h-9 w-64 bg-slate-50 border border-slate-200 rounded-full pl-9 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-300 transition-all text-slate-900 placeholder:text-slate-400"
                    />
                </div>

                <button className="relative p-2 text-slate-400 hover:text-slate-600 transition-colors rounded-full hover:bg-slate-50">
                    <Bell className="w-5 h-5" />
                    <span className="absolute top-2 right-2.5 w-2 h-2 rounded-full bg-blue-600 border border-white" />
                </button>

                <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-100 to-blue-200 border border-blue-300 flex items-center justify-center text-blue-700 font-semibold cursor-pointer shadow-sm">
                    A
                </div>
            </div>
        </header>
    );
}
