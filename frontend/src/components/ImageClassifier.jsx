import React, { useState, useCallback } from 'react';
import { predictImage } from '../api/cifarApi';

const CLASS_REFERENCE = [
  { name: 'airplane', icon: '✈️' },
  { name: 'automobile', icon: '🚗' },
  { name: 'bird', icon: '🐦' },
  { name: 'cat', icon: '🐱' },
  { name: 'deer', icon: '🦌' },
  { name: 'dog', icon: '🐶' },
  { name: 'frog', icon: '🐸' },
  { name: 'horse', icon: '🐴' },
  { name: 'ship', icon: '🚢' },
  { name: 'truck', icon: '🚚' },
];

const ProbabilityBar = ({ label, probability, colorClass }) => (
  <div className="mb-2">
    <div className="flex justify-between text-xs mb-1 text-gray-300">
      <span>{label}</span>
      <span>{(probability * 100).toFixed(1)}%</span>
    </div>
    <div className="w-full bg-gray-700/50 rounded-full h-1.5">
      <div
        className={`h-1.5 rounded-full ${colorClass}`}
        style={{ width: `${probability * 100}%` }}
      ></div>
    </div>
  </div>
);

const ResultCard = ({ title, result, isWinner, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex-1 min-w-[300px] p-6 rounded-2xl bg-white/5 border border-white/10 animate-pulse">
        <div className="h-6 w-24 bg-white/10 rounded mb-4"></div>
        <div className="h-10 w-full bg-white/10 rounded mb-6"></div>
        <div className="space-y-3">
          <div className="h-4 w-full bg-white/10 rounded"></div>
          <div className="h-4 w-full bg-white/10 rounded"></div>
          <div className="h-4 w-full bg-white/10 rounded"></div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  // Get top 3 probabilities
  const top3 = Object.entries(result.probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);

  return (
    <div
      className={`flex-1 min-w-[300px] p-6 rounded-2xl transition-all duration-500 backdrop-blur-md ${
        isWinner
          ? 'bg-green-500/10 border-green-500/50 shadow-[0_0_20px_rgba(34,197,94,0.2)] ring-1 ring-green-500/50'
          : 'bg-white/5 border-white/10 opacity-60 grayscale-[0.3]'
      }`}
    >
      <h3 className={`text-sm font-semibold mb-2 uppercase tracking-wider ${isWinner ? 'text-green-400' : 'text-gray-400'}`}>
        {title} {isWinner && '🏆'}
      </h3>
      <div className="mb-4">
        <span className="inline-block px-4 py-1.5 rounded-full bg-white/10 text-white text-xl font-bold mb-1 border border-white/10">
          {result.class}
        </span>
        <p className="text-gray-400 text-sm">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
      </div>
      
      <div className="mt-6">
        <p className="text-xs font-medium text-gray-500 mb-3 uppercase tracking-widest">Distribution</p>
        {top3.map(([label, prob]) => (
          <ProbabilityBar
            key={label}
            label={label}
            probability={prob}
            colorClass={isWinner ? 'bg-green-500' : 'bg-blue-500'}
          />
        ))}
      </div>
    </div>
  );
};

const ImageClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setLoading(true);

    try {
      const data = await predictImage(file);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to classify image. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 text-white min-h-screen">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-black mb-3 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          CIFAR-10 Visual Analyzer
        </h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Deep learning performance comparison: Artificial Neural Network vs Multiscale Convolutional Neural Network.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="flex flex-col items-center mb-12">
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`relative group w-full max-w-xl p-8 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer ${
            isDragging
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-white/20 bg-white/5 hover:border-white/40'
          }`}
          onClick={() => document.getElementById('fileInput').click()}
        >
          <input
            id="fileInput"
            type="file"
            className="hidden"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          
          <div className="flex flex-col items-center text-center">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="w-32 h-32 object-cover rounded-xl shadow-2xl mb-4 border-2 border-white/20"
              />
            ) : (
              <div className="w-20 h-20 flex items-center justify-center rounded-2xl bg-white/5 mb-4 text-4xl">
                📸
              </div>
            )}
            <p className="text-lg font-medium text-gray-200">
              {selectedFile ? selectedFile.name : 'Drop image here or click to upload'}
            </p>
            <p className="text-sm text-gray-500 mt-1">Supports JPG, PNG, WEBP (CIFAR-10: 32x32)</p>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 text-red-400 rounded-xl text-sm">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {(loading || results) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16 items-stretch">
          <ResultCard
            title="ANN (3-Layers)"
            result={results?.ann}
            isWinner={results?.winner === 'ann'}
            isLoading={loading}
          />
          <ResultCard
            title="CNN (Multiscale)"
            result={results?.cnn}
            isWinner={results?.winner === 'cnn'}
            isLoading={loading}
          />
        </div>
      )}

      {/* Class Reference Grid */}
      <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-sm">
        <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-6 text-center">
          CIFAR-10 Class Reference
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
          {CLASS_REFERENCE.map((item) => (
            <div
              key={item.name}
              className="flex flex-col items-center p-4 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors"
            >
              <span className="text-2xl mb-2">{item.icon}</span>
              <span className="text-xs font-medium text-gray-400 capitalize">{item.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImageClassifier;
