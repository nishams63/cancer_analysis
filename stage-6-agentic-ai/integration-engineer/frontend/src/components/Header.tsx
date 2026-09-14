import React from 'react';

export const Header: React.FC = () => (
  <header className="border-b border-slate-800 bg-slate-950 px-6 py-4 flex items-center justify-between">
    <div className="flex items-center space-x-3">
      <div className="h-8 w-8 rounded-lg bg-emerald-500 flex items-center justify-center font-bold text-slate-950 text-lg">
        A
      </div>
      <div>
        <h1 className="text-lg font-bold text-slate-100 leading-none">AADA Dashboard</h1>
        <span className="text-xs text-slate-400">Autonomous AI Data Analyst — Stage 06</span>
      </div>
    </div>
    <div className="flex items-center space-x-4 text-xs font-mono text-slate-400">
      <span className="inline-flex items-center gap-1.5 py-1 px-2.5 rounded-full bg-slate-800 text-emerald-400 border border-emerald-500/20">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
        API Connected
      </span>
    </div>
  </header>
);
