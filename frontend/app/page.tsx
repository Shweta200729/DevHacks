"use client";
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";
import { PlayCircle, ShieldAlert, Cpu, Database } from "lucide-react";

interface MetricsData {
  current_version: number;
  evaluations: any[];
  aggregations: any[];
  pending_queue_size: number;
}

export default function Dashboard() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [simName, setSimName] = useState("Edge-1");
  const [isMalicious, setIsMalicious] = useState(false);

  const fetchMetrics = async () => {
    try {
      // Assuming FastAPI runs on 8000
      const res = await fetch("http://localhost:8000/metrics");
      if (res.ok) {
        const json = await res.json();
        // Reverse arrays so oldest is first for Recharts left-to-right plotting
        json.evaluations = json.evaluations.reverse();
        json.aggregations = json.aggregations.reverse();
        setData(json);
      }
    } catch (e) {
      console.error("Failed fetching metrics", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleSimulate = async () => {
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
      alert(`Simulation started for ${simName} (Check terminal logs for rejection status)`);
    } catch (e) {
      alert("Simulation attempt failed");
    }
  };

  const calculateTotalClients = () => {
    if (!data?.aggregations) return { acc: 0, rej: 0 };
    let acc = 0;
    let rej = 0;
    data.aggregations.forEach(a => {
      acc += a.total_accepted;
      rej += a.total_rejected;
    });
    return { acc, rej };
  };

  const stats = calculateTotalClients();

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans">
      <header className="mb-10 flex items-center justify-between border-b border-gray-800 pb-6">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            AFL Production Real-Time Dashboard
          </h1>
          <p className="text-gray-400 mt-2">Asynchronous Federated Learning</p>
        </div>
        <div className="flex gap-4 items-center">
          <div className="bg-gray-900 border border-gray-700 px-4 py-2 rounded-xl flex items-center gap-2 shadow-sm">
            <Database className="w-5 h-5 text-indigo-400" />
            <span className="font-semibold text-gray-200">
              Live Version: v{data?.current_version || 0}
            </span>
          </div>
          <div className="bg-gray-900 border border-gray-700 px-4 py-2 rounded-xl flex items-center gap-2 shadow-sm">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <span className="font-semibold text-gray-200">
              Queue: {data?.pending_queue_size || 0} Clients
            </span>
          </div>
        </div>
      </header>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl">
          <h3 className="text-gray-400 text-sm font-medium mb-1 uppercase tracking-wider">Accepted Clients</h3>
          <p className="text-4xl font-bold text-emerald-400">{stats.acc}</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl">
          <h3 className="text-gray-400 text-sm font-medium mb-1 uppercase tracking-wider">Rejected Clients (Malicious)</h3>
          <p className="text-4xl font-bold text-red-400">{stats.rej}</p>
        </div>
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl">
          <h3 className="text-gray-400 text-sm font-medium mb-1 uppercase tracking-wider">Current Accuracy</h3>
          <p className="text-4xl font-bold text-blue-400">
            {data?.evaluations && data.evaluations.length > 0
              ? (data.evaluations[data.evaluations.length - 1].accuracy * 100).toFixed(2) + "%"
              : "N/A"
            }
          </p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
        {/* Accuracy Chart */}
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl">
          <h2 className="text-xl font-bold mb-6 text-gray-200">Global Model Accuracy over Time</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.evaluations || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="version_id" stroke="#9CA3AF" tickFormatter={(t) => `v${t}`} />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151' }}
                  itemStyle={{ color: '#E5E7EB' }}
                />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#60A5FA" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} name="Validation Acc" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss Chart */}
        <div className="bg-gray-900 border border-gray-800 p-6 rounded-2xl">
          <h2 className="text-xl font-bold mb-6 text-gray-200">Global Model Loss over Time</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.evaluations || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="version_id" stroke="#9CA3AF" tickFormatter={(t) => `v${t}`} />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151' }}
                  itemStyle={{ color: '#E5E7EB' }}
                />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#F472B6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} name="Cross Entropy Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Admin Simulation Controller */}
      <div className="bg-indigo-950/30 border border-indigo-900/50 p-6 rounded-2xl mt-10">
        <h2 className="text-xl font-bold mb-4 text-indigo-300 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5" />
          Simulation Controller
        </h2>
        <p className="text-gray-400 mb-6 max-w-2xl">
          Inject authentic client training updates directly into the FastAPI endpoint.
          Normal clients train a CNN randomly over an isolated MNIST batch.
          Malicious clients apply harsh noise logic generating high $L^2$ bounds.
        </p>

        <div className="flex flex-wrap gap-4 items-end">
          <div className="flex flex-col gap-2">
            <label className="text-sm text-gray-400 font-semibold uppercase tracking-wide">Client Node Alias</label>
            <input
              className="bg-gray-900 border border-gray-700 text-white px-4 py-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 w-64"
              value={simName}
              onChange={(e) => setSimName(e.target.value)}
            />
          </div>

          <div className="flex items-center gap-3 bg-gray-900 border border-gray-700 px-4 py-2.5 rounded-lg">
            <input
              type="checkbox"
              checked={isMalicious}
              onChange={(e) => setIsMalicious(e.target.checked)}
              className="w-5 h-5 rounded border-gray-700 text-indigo-500 focus:ring-indigo-500 bg-gray-800"
            />
            <label className="text-red-400 font-medium">Inject Data Poisoning Payload</label>
          </div>

          <button
            onClick={handleSimulate}
            className="bg-indigo-600 hover:bg-indigo-700 transition-colors text-white px-6 py-2.5 rounded-lg font-semibold flex items-center gap-2 shadow-lg shadow-indigo-500/20"
          >
            <PlayCircle className="w-5 h-5" />
            Fire Simulation Round
          </button>
        </div>
      </div>
    </div>
  );
}
