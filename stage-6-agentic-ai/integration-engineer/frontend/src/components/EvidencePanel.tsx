import React from 'react';
import { AnalysisResult } from '../types';

interface EvidencePanelProps {
  analysis?: AnalysisResult;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({ analysis }) => {
  if (!analysis || !analysis.evidence || analysis.evidence.length === 0) return null;

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">4. Evidence Panel</h2>
      <div className="space-y-3">
        {analysis.evidence.map((item, idx) => (
          <div key={idx} className="bg-slate-900 border border-slate-800 rounded-lg p-3">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-mono text-slate-400">Data Point #{idx + 1}</span>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                item.classification === 'OBSERVED FACT' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-blue-950 text-blue-400 border border-blue-800'
              }`}>
                {item.classification}
              </span>
            </div>
            <pre className="text-xs font-mono text-slate-200 overflow-x-auto">{JSON.stringify(item.data, null, 2)}</pre>
          </div>
        ))}
      </div>
    </div>
  );
};
