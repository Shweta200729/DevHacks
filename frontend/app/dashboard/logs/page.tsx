import { TracingBeam } from "@/components/ui/tracing-beam";
import { TerminalSquare } from "lucide-react";

export default function LogsPage() {
    const logEntries = [
        { time: "14:02:11.432", level: "info", title: "Federation Round Started", msg: "Initializing Round 45. Waiting for minimum 100 client gradients.", type: "system" },
        { time: "14:02:45.102", level: "info", title: "Client Update Rcvd", msg: "Client [0A3X1] submitted 4MB gradient package. Added to queue.", type: "client" },
        { time: "14:02:46.883", level: "info", title: "Client Update Rcvd", msg: "Client [B89Q2] submitted 4MB gradient package. Added to queue.", type: "client" },
        { time: "14:02:51.011", level: "warn", title: "Update Rejected", msg: "Client [X99Q0] rejected. Out of bounds penalty (L2 Norm: 24.5 > 10.0).", type: "security" },
        { time: "14:04:15.900", level: "info", title: "Threshold Reached", msg: "100 updates queued. Initiating aggregation protocol.", type: "system" },
        { time: "14:04:16.050", level: "info", title: "Aggregation Started", msg: "Applying Trimmed Mean (discard rate: 10%). Resolving 90 gradients.", type: "compute" },
        { time: "14:04:45.312", level: "success", title: "Global Model Compiled", msg: "Version v2.0.5 successfully built. Validation ACC: 94.6%.", type: "system" },
        { time: "14:05:00.000", level: "info", title: "Federation Round Ended", msg: "Dispatching v2.0.5 weights to active clients. Entering idle state.", type: "system" },
    ];

    return (
        <div className="flex flex-col gap-8 pb-32">
            <div className="flex flex-col gap-2 mb-8">
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-3">
                    <TerminalSquare className="w-8 h-8 text-blue-600" /> System Events
                </h2>
                <p className="text-slate-500">Live, chronological event stream from the aggregation backend.</p>
            </div>

            <TracingBeam className="pl-6">
                <div className="flex flex-col gap-4 w-full">
                    {logEntries.map((log, i) => (
                        <div key={i} className="bg-slate-50/80 border border-slate-200 rounded-lg p-5 font-mono text-sm relative group hover:bg-white hover:shadow-md transition-all duration-300">
                            {/* Event type marker */}
                            <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg ${log.level === 'warn' ? 'bg-amber-400' :
                                    log.level === 'success' ? 'bg-green-500' :
                                        log.type === 'compute' ? 'bg-purple-500' :
                                            'bg-blue-500'
                                }`} />

                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-3">
                                    <span className={`font-semibold bg-white px-2 py-0.5 rounded shadow-sm border ${log.level === 'warn' ? 'text-amber-700 border-amber-200' :
                                            log.level === 'success' ? 'text-green-700 border-green-200' :
                                                'text-blue-700 border-blue-200'
                                        }`}>
                                        {log.title}
                                    </span>
                                </div>
                                <span className="text-slate-400 group-hover:text-blue-500 transition-colors">[{log.time}]</span>
                            </div>

                            <div className="pl-1 text-slate-600">
                                <span className="text-slate-400 mr-2">&gt;</span>{log.msg}
                            </div>

                            {log.level === 'warn' && (
                                <div className="mt-3 bg-amber-50 border border-amber-200 text-amber-800 p-2 text-xs rounded">
                                    ! SECURITY TRACE: Malicious payload signature detected in vector [1024:2048].
                                </div>
                            )}
                        </div>
                    ))}

                    <div className="flex items-center justify-center py-8">
                        <div className="animate-pulse flex items-center gap-2 text-slate-400 text-sm font-mono">
                            <span className="w-2 h-2 rounded-full bg-slate-400" />
                            Waiting for events...
                        </div>
                    </div>
                </div>
            </TracingBeam>
        </div>
    );
}
