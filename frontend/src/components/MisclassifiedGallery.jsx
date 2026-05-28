import React, { useState, useEffect } from 'react';
import { getMisclassified } from '../api/cifarApi';

const SkeletonCard = () => (
  <div className="p-4 rounded-2xl bg-white/5 border border-white/10 animate-pulse">
    <div className="w-full aspect-square bg-white/10 rounded-lg mb-3"></div>
    <div className="space-y-2">
      <div className="h-4 w-20 bg-white/10 rounded"></div>
      <div className="h-4 w-24 bg-white/10 rounded"></div>
    </div>
  </div>
);

const MisclassifiedGallery = () => {
  const [modelType, setModelType] = useState('ann');
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchErrors = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getMisclassified(modelType);
        // Take only the first 10 for the 2x5 grid as per requirements
        setSamples(data.slice(0, 10));
      } catch (err) {
        setError(err.message || `Failed to load ${modelType.toUpperCase()} errors`);
      } finally {
        setLoading(false);
      }
    };
    fetchErrors();
  }, [modelType]);

  const getConfusionInsight = () => {
    if (!samples.length) return null;
    
    const pairs = samples.map(s => `${s.true_class}->${s.predicted_class}`);
    const counts = pairs.reduce((acc, p) => {
      acc[p] = (acc[p] || 0) + 1;
      return acc;
    }, {});
    
    const mostCommon = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    const [trueClass, predClass] = mostCommon.split('->');
    
    return { trueClass, predClass };
  };

  const insight = getConfusionInsight();

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Header & Toggle */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <h2 className="text-2xl font-black text-white">Error Analysis</h2>
          <p className="text-gray-400 text-sm">Visualizing samples where the model's prediction failed</p>
        </div>
        
        <div className="flex p-1 bg-white/5 border border-white/10 rounded-xl backdrop-blur-md">
          <button
            onClick={() => setModelType('ann')}
            className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${
              modelType === 'ann' 
                ? 'bg-blue-500 text-white shadow-lg' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            ANN Errors
          </button>
          <button
            onClick={() => setModelType('cnn')}
            className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${
              modelType === 'cnn' 
                ? 'bg-purple-500 text-white shadow-lg' 
                : 'text-gray-400 hover:text-white'
            }`}
          >
            CNN Errors
          </button>
        </div>
      </div>

      {/* Grid Display */}
      {error ? (
        <div className="p-12 text-center rounded-3xl bg-red-500/5 border border-red-500/20 text-red-400">
          <p>⚠️ {error}</p>
          <p className="text-xs mt-2 opacity-60 italic">Make sure the misclassified metrics are generated in the backend</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-6">
          {loading ? (
            Array(10).fill(0).map((_, i) => <SkeletonCard key={i} />)
          ) : (
            samples.map((sample, idx) => (
              <div 
                key={idx} 
                className="group p-4 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm hover:border-white/30 transition-all hover:-translate-y-1"
              >
                <div className="relative w-full aspect-square bg-slate-900 rounded-lg overflow-hidden mb-4 ring-1 ring-white/10">
                  {/* Using a placeholder since we don't have individual image URLs, 
                      but in a real app this would be a link to the static assets */}
                  <div className="w-full h-full flex items-center justify-center text-3xl opacity-40 grayscale group-hover:grayscale-0 transition-all">
                    🖼️
                  </div>
                  <div className="absolute top-2 right-2 px-2 py-0.5 bg-black/60 backdrop-blur-md rounded text-[10px] text-gray-400">
                    ID: {sample.index}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-gray-500 uppercase tracking-tighter">True</span>
                    <span className="px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 text-[11px] font-bold border border-emerald-500/30">
                      {sample.true_class}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-gray-500 uppercase tracking-tighter">Pred</span>
                    <span className="px-2 py-0.5 rounded-md bg-red-500/20 text-red-400 text-[11px] font-bold border border-red-500/30">
                      {sample.predicted_class}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Insight Caption */}
      {!loading && !error && insight && (
        <div className="text-center p-6 rounded-2xl bg-white/5 border border-white/5 backdrop-blur-sm">
          <p className="text-sm text-gray-400 italic">
            "Model confused <span className="text-emerald-400 font-bold not-italic">{insight.trueClass}</span> for <span className="text-red-400 font-bold not-italic">{insight.predClass}</span> most often in this subset."
          </p>
        </div>
      )}
    </div>
  );
};

export default MisclassifiedGallery;
