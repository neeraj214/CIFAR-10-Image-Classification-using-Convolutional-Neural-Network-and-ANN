import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getComparison } from '../api/cifarApi';

const ComparisonDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchComparison = async () => {
      try {
        const result = await getComparison();
        setData(result);
      } catch (err) {
        setError(err.message || 'Failed to load comparison data');
      } finally {
        setLoading(false);
      }
    };
    fetchComparison();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-500/10 border border-red-500/50 rounded-2xl text-red-400 text-center mx-auto max-w-2xl">
        <p className="font-bold mb-2">Error Loading Dashboard</p>
        <p className="text-sm">{error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg transition-colors text-xs"
        >
          Try Again
        </button>
      </div>
    );
  }

  // Prepare chart data
  const chartData = [
    { name: 'Accuracy', ann: data.ann.test_accuracy, cnn: data.cnn.test_accuracy },
    { name: 'Precision', ann: data.ann.precision, cnn: data.cnn.precision },
    { name: 'Recall', ann: data.ann.recall, cnn: data.cnn.recall },
    { name: 'F1-Score', ann: data.ann.f1, cnn: data.cnn.f1 },
  ];

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-12 animate-in fade-in duration-700">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-black bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          Model Performance Analytics
        </h2>
        <p className="text-gray-400 text-sm">Deep dive into ANN vs CNN metrics and architecture</p>
      </div>

      {/* Section 1: Charts */}
      <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl shadow-2xl">
        <h3 className="text-lg font-bold text-gray-200 mb-8 flex items-center gap-2">
          <span className="p-2 bg-blue-500/20 rounded-lg text-blue-400 text-sm">📊</span>
          Metric Comparison
        </h3>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                itemStyle={{ fontSize: '12px' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar dataKey="ann" name="ANN (Dense)" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={40} />
              <Bar dataKey="cnn" name="CNN (Multiscale)" fill="#a855f7" radius={[4, 4, 0, 0]} barSize={40} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Section 2: Summary Table */}
      <div className="overflow-hidden rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl shadow-2xl">
        <div className="p-6 border-b border-white/10 bg-white/5">
          <h3 className="text-lg font-bold text-gray-200 flex items-center gap-2">
            <span className="p-2 bg-emerald-500/20 rounded-lg text-emerald-400 text-sm">📋</span>
            Detailed Summary
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-white/5 text-gray-400 text-xs uppercase tracking-widest">
                <th className="px-6 py-4 font-semibold">Model</th>
                <th className="px-6 py-4 font-semibold text-right">Params</th>
                <th className="px-6 py-4 font-semibold text-right">Test Accuracy</th>
                <th className="px-6 py-4 font-semibold text-right">Training Time</th>
                <th className="px-6 py-4 font-semibold text-right">F1 Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {[data.ann, data.cnn].map((model, idx) => (
                <tr key={idx} className="hover:bg-white/5 transition-colors">
                  <td className="px-6 py-5 font-medium text-gray-200">{model.model}</td>
                  <td className="px-6 py-5 text-right text-gray-400 font-mono text-sm">{model.params.toLocaleString()}</td>
                  <td className="px-6 py-5 text-right">
                    <span className="px-2 py-1 rounded bg-blue-500/10 text-blue-400 text-sm font-bold">
                      {(model.test_accuracy * 100).toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-6 py-5 text-right text-gray-400 text-sm">{model.training_time.toFixed(1)}s</td>
                  <td className="px-6 py-5 text-right text-gray-400 text-sm">{(model.f1).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 3: Architecture Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* ANN Card */}
        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl hover:border-blue-500/30 transition-all group">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 bg-blue-500/20 rounded-2xl text-blue-400 group-hover:scale-110 transition-transform">🧠</div>
            <div>
              <h4 className="font-bold text-gray-200">ANN Architecture</h4>
              <p className="text-xs text-gray-500">Dense Neural Network</p>
            </div>
          </div>
          <ul className="space-y-4">
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Input Layer</span>
              <span className="text-blue-400 font-mono">3072 neurons</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Hidden Layer 1</span>
              <span className="text-blue-400 font-mono">512 neurons</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Hidden Layer 2</span>
              <span className="text-blue-400 font-mono">256 neurons</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Hidden Layer 3</span>
              <span className="text-blue-400 font-mono">128 neurons</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Output Layer</span>
              <span className="text-blue-400 font-mono">10 classes</span>
            </li>
          </ul>
        </div>

        {/* CNN Card */}
        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl hover:border-purple-500/30 transition-all group">
          <div className="flex items-center gap-4 mb-6">
            <div className="p-3 bg-purple-500/20 rounded-2xl text-purple-400 group-hover:scale-110 transition-transform">👁️</div>
            <div>
              <h4 className="font-bold text-gray-200">CNN Architecture</h4>
              <p className="text-xs text-gray-500">Multiscale Convolutional Network</p>
            </div>
          </div>
          <ul className="space-y-4">
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Input Volume</span>
              <span className="text-purple-400 font-mono">32x32x3</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Conv Layer 1</span>
              <span className="text-purple-400 font-mono">32 filters (3x3)</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Conv Layer 2</span>
              <span className="text-purple-400 font-mono">64 filters (5x5)</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Conv Layer 3</span>
              <span className="text-purple-400 font-mono">128 filters (7x7)</span>
            </li>
            <li className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 text-sm">
              <span className="text-gray-400">Dense Layer</span>
              <span className="text-purple-400 font-mono">256 units</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ComparisonDashboard;
