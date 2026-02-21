"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabaseClient";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [session, setSession] = useState<any>(null);

    useEffect(() => {
        let mounted = true;

        async function checkSession() {
            try {
                const { data: { session } } = await supabase.auth.getSession();
                if (mounted) {
                    if (!session) {
                        router.push('/login');
                    } else {
                        setSession(session);
                    }
                }
            } catch (err) {
                console.error("Session check failed", err);
                if (mounted) router.push('/login');
            } finally {
                if (mounted) setLoading(false);
            }
        }

        checkSession();

        const { data: authListener } = supabase.auth.onAuthStateChange(
            (event, session) => {
                if (mounted) {
                    if (!session) {
                        router.push('/login');
                    } else {
                        setSession(session);
                    }
                }
            }
        );

        return () => {
            mounted = false;
            authListener.subscription.unsubscribe();
        };
    }, [router]);

    if (loading) {
        return (
            <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 font-sans">
                <span className="text-slate-500 font-medium tracking-wide animate-pulse">Authenticating...</span>
            </div>
        );
    }

    if (!session) {
        return null; // Will redirect in useEffect
    }

    return <>{children}</>;
}
