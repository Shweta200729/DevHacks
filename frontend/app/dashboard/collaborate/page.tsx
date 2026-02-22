"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Users2, CheckCircle2, XCircle, Search, UserPlus, Zap, Clock, BoxIcon } from "lucide-react";
import {
    fetchCollabUsers,
    fetchMyCollabSessions,
    sendCollabRequest,
    respondToCollabRequest,
    CollabUser,
    CollabSession
} from "@/lib/api";
import { supabase } from "@/lib/supabaseClient";

export default function CollaboratePage() {
    // We mock a logged-in user for this demo. Normally extracted from JWT.
    // For DevHacks, we'll act as User 1 if no context is found, but normally this would be dynamic.
    const [currentUserId, setCurrentUserId] = useState<number>(1);

    const [allUsers, setAllUsers] = useState<CollabUser[]>([]);
    const [sessions, setSessions] = useState<CollabSession[]>([]);
    const [search, setSearch] = useState("");

    const loadData = useCallback(async () => {
        const [u, s] = await Promise.all([
            fetchCollabUsers(),
            fetchMyCollabSessions(currentUserId),
        ]);
        setAllUsers(u.filter(user => user.id !== currentUserId));
        setSessions(s);
    }, [currentUserId]);

    useEffect(() => {
        loadData();

        // Setup real-time listener for incoming requests / status changes
        const channel = supabase.channel('collab-changes')
            .on(
                'postgres_changes',
                { event: '*', schema: 'public', table: 'collab_sessions' },
                () => {
                    loadData(); // refresh when any session changes
                }
            )
            .subscribe();

        return () => { supabase.removeChannel(channel); };
    }, [loadData]);


    const handleRequest = async (toUserId: number) => {
        await sendCollabRequest(currentUserId, toUserId, "Let's train together!");
        await loadData();
    };

    const handleRespond = async (sessionId: string, action: "accept" | "reject" | "cancel") => {
        await respondToCollabRequest(currentUserId, sessionId, action);
        await loadData();
    };

    // Derived state
    const pendingInbox = sessions.filter(s => s.status === "pending" && s.recipient_id === currentUserId);
    const pendingSent = sessions.filter(s => s.status === "pending" && s.requester_id === currentUserId);
    const activeSessions = sessions.filter(s => s.status === "active" || s.status === "completed");

    const filteredUsers = allUsers.filter(u =>
        u.name.toLowerCase().includes(search.toLowerCase()) ||
        u.email.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="flex flex-col gap-8 pb-10 max-w-5xl">
            <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-3">
                    <Users2 className="w-8 h-8 text-indigo-600" />
                    Collaboration Hub
                </h2>
                <p className="text-slate-500">
                    Discover researchers, send invites, and train federated models together safely.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* LEFT COL: Inbox & Active */}
                <div className="lg:col-span-2 flex flex-col gap-6">

                    {/* INBOX */}
                    <Card className="bg-white border-slate-200 shadow-sm border-t-4 border-t-amber-400">
                        <CardHeader className="pb-4">
                            <CardTitle className="text-lg font-bold flex items-center justify-between">
                                <span className="flex items-center gap-2">
                                    <Clock className="w-5 h-5 text-amber-500" />
                                    Pending Requests
                                </span>
                                {pendingInbox.length > 0 && (
                                    <span className="bg-amber-100 text-amber-800 text-xs font-black px-2 py-0.5 rounded-full">
                                        {pendingInbox.length} NEW
                                    </span>
                                )}
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="flex flex-col gap-3">
                            {pendingInbox.length === 0 ? (
                                <p className="text-sm text-slate-400 italic">No incoming requests.</p>
                            ) : (
                                pendingInbox.map(s => (
                                    <div key={s.id} className="flex items-center justify-between p-4 rounded-xl border border-amber-100 bg-amber-50/50">
                                        <div>
                                            <p className="font-bold text-slate-900">{s.partner_name}</p>
                                            <p className="text-xs text-slate-500 mt-1">"{s.message || "Wants to collaborate"}"</p>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <button
                                                onClick={() => handleRespond(s.id, "accept")}
                                                className="px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm font-bold rounded-lg transition-colors shadow-sm"
                                            >
                                                Accept
                                            </button>
                                            <button
                                                onClick={() => handleRespond(s.id, "reject")}
                                                className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                                            >
                                                <XCircle className="w-5 h-5" />
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </CardContent>
                    </Card>

                    {/* ACTIVE SESSIONS */}
                    <Card className="bg-white border-slate-200 shadow-sm border-t-4 border-t-indigo-500">
                        <CardHeader className="pb-4">
                            <CardTitle className="text-lg font-bold flex items-center gap-2">
                                <Zap className="w-5 h-5 text-indigo-500" />
                                Active Collaborations
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="flex flex-col gap-4">
                            {activeSessions.length === 0 ? (
                                <p className="text-sm text-slate-400 italic">No active sessions. Invite someone to start!</p>
                            ) : (
                                activeSessions.map(s => (
                                    <div key={s.id} className="p-5 rounded-2xl border border-indigo-100 bg-gradient-to-b from-white to-indigo-50/30 shadow-sm flex flex-col gap-4">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-black text-lg border border-indigo-200">
                                                    {s.partner_name?.charAt(0).toUpperCase()}
                                                </div>
                                                <div>
                                                    <h4 className="font-extrabold text-slate-900">{s.partner_name}</h4>
                                                    <p className="text-xs font-semibold text-indigo-600 uppercase tracking-widest">{s.status}</p>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => handleRespond(s.id, "cancel")}
                                                className="text-xs font-bold text-slate-400 hover:text-red-600 transition-colors"
                                            >
                                                Leave
                                            </button>
                                        </div>

                                        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
                                            <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg border border-slate-100">
                                                <div className="text-sm font-semibold text-slate-700 font-mono flex items-center gap-2">
                                                    <BoxIcon className="w-4 h-4 text-slate-400" />
                                                    Session ID: {s.id.split('-')[0]}
                                                </div>
                                                {s.shared_version_id && (
                                                    <div className="text-xs font-bold bg-green-100 text-green-700 px-2 py-1 rounded">
                                                        Model v{s.shared_version_id} Ready
                                                    </div>
                                                )}
                                            </div>

                                            <p className="text-sm text-slate-500">
                                                Both parties must submit updates using this Session ID before the shared global model aggregates.
                                            </p>

                                            <button className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-indigo-200 transition-all flex items-center justify-center gap-2">
                                                <Zap className="w-4 h-4" />
                                                Upload Data & Train Together
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </CardContent>
                    </Card>

                </div>


                {/* RIGHT COL: Discover */}
                <div className="flex flex-col gap-6">
                    <Card className="bg-white border-slate-200 shadow-sm overflow-hidden flex flex-col h-[600px]">
                        <CardHeader className="border-b border-slate-100 bg-slate-50/50 pb-4">
                            <CardTitle className="text-lg font-bold">Discover Network</CardTitle>
                            <CardDescription>Find other registered researchers.</CardDescription>

                            <div className="relative mt-4">
                                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                                <input
                                    type="text"
                                    placeholder="Search names..."
                                    value={search}
                                    onChange={e => setSearch(e.target.value)}
                                    className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow"
                                />
                            </div>
                        </CardHeader>

                        <CardContent className="p-0 overflow-y-auto flex-1">
                            <div className="divide-y divide-slate-100">
                                {filteredUsers.map(u => {
                                    const isPendingSent = pendingSent.some(s => s.partner_id === u.id);
                                    const isPendingInbox = pendingInbox.some(s => s.partner_id === u.id);
                                    const isActive = activeSessions.some(s => s.partner_id === u.id);

                                    return (
                                        <div key={u.id} className="p-4 hover:bg-slate-50 transition-colors flex items-center justify-between group">
                                            <div>
                                                <p className="font-bold text-slate-800 text-sm">{u.name}</p>
                                                <p className="text-xs text-slate-400 truncate">{u.email}</p>
                                            </div>

                                            {isActive ? (
                                                <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-1 rounded-md">Active</span>
                                            ) : isPendingSent ? (
                                                <span className="text-xs font-bold text-amber-600 bg-amber-50 px-2 py-1 rounded-md">Requested</span>
                                            ) : isPendingInbox ? (
                                                <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded-md">Review Inbox</span>
                                            ) : (
                                                <button
                                                    onClick={() => handleRequest(u.id)}
                                                    className="p-1.5 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                                                    title="Send Invite"
                                                >
                                                    <UserPlus className="w-5 h-5" />
                                                </button>
                                            )}
                                        </div>
                                    );
                                })}
                                {filteredUsers.length === 0 && (
                                    <div className="p-8 text-center text-sm text-slate-400">
                                        No users found.
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>

            </div>
        </div>
    );
}
