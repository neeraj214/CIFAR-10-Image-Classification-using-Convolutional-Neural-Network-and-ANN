import React, { useState } from 'react';
import ImageClassifier from './components/ImageClassifier';
import ComparisonDashboard from './components/ComparisonDashboard';
import MisclassifiedGallery from './components/MisclassifiedGallery';

const TABS = [
  { id: 'classify', label: 'Classify', icon: '📸' },
  { id: 'benchmark', label: 'Benchmark', icon: '📊' },
  { id: 'errors', label: 'Errors', icon: '🔍' },
];

function App() {
  const [activeTab, setActiveTab] = useState('classify');

  const renderContent = () => {
    switch (activeTab) {
      case 'classify':
        return <ImageClassifier />;
      case 'benchmark':
        return <ComparisonDashboard />;
      case 'errors':
        return <MisclassifiedGallery />;
      default:
        return <ImageClassifier />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-blue-500/30">
      {/* Navigation Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-slate-950/80 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="text-center md:text-left">
            <h1 className="text-2xl font-black bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
              CIFAR-10 Classifier
            </h1>
            <p className="text-xs text-gray-500 font-medium uppercase tracking-widest mt-0.5">
              CNN (3×3, 5×5, 7×7) vs ANN Deep Compare
            </p>
          </div>

          <nav className="flex p-1 bg-white/5 rounded-2xl border border-white/10">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative px-6 py-2 rounded-xl text-sm font-bold transition-all duration-300 flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'text-white'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {activeTab === tab.id && (
                  <div className="absolute inset-0 rounded-xl border-2 border-transparent bg-gradient-to-r from-blue-500 to-purple-500 [mask-image:linear-gradient(white,white)_padding-box,linear-gradient(white,white)]" />
                )}
                {activeTab === tab.id && (
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/10 to-purple-500/10" />
                )}
                <span className="relative z-10">{tab.icon}</span>
                <span className="relative z-10">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="relative py-12">
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[128px] -z-10" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-[128px] -z-10" />
        
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="py-12 border-t border-white/5 text-center">
        <p className="text-gray-600 text-xs">
          Built for Deep Learning Performance Analysis • CIFAR-10 Dataset
        </p>
      </footer>
    </div>
  );
}

export default App;
