import React from 'react';
import { AnalysisResult } from '../types';

interface ResultPanelProps {
  analysis?: AnalysisResult;
}

export const ResultPanel: React.FC<ResultPanelProps> = ({ analysis }) => {
  if (!analysis) return null;

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">6. Final Analytical Result</h2>
      <div className="mb-4">
        <h3 className="text-xs font-mono text-slate-500 mb-1 uppercase">Executive Summary</h3>
        <p className="text-sm text-slate-200 leading-relaxed bg-slate-900 p-3 rounded-lg border border-slate-800">
          {analysis.summary}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <h3 className="text-xs font-mono text-slate-500 mb-2 uppercase">Key Findings</h3>
          <ul className="space-y-1.5">
            {analysis.findings.map((f, i) => (
              <li key={i} className="text-xs text-slate-300 bg-slate-900 px-3 py-2 rounded border border-slate-800">
                <strong>{f.cause}</strong> {f.score !== undefined && `(score: ${f.score})`}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="text-xs font-mono text-slate-500 mb-2 uppercase">Recommendations</h3>
          <ul className="space-y-1.5">
            {analysis.recommendations.map((r, i) => (
              <li key={i} className="text-xs text-slate-300 bg-slate-900 px-3 py-2 rounded border border-slate-800">
                {i + 1}. {r}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
