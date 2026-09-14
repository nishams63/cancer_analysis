import React from 'react';
import { EvaluationData } from '../types';

interface EvaluationCardProps {
  evaluation?: EvaluationData;
}

export const EvaluationCard: React.FC<EvaluationCardProps> = ({ evaluation }) => {
  if (!evaluation || !evaluation.evaluated) return null;

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">5. Evaluation Scorecard</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-xs text-slate-500 font-mono">PROCESS QUALITY</div>
          <div className="text-lg font-bold text-emerald-400 font-mono">
            {evaluation.process_score !== undefined ? `${Math.round(evaluation.process_score * 100)}%` : '--'}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">OUTCOME QUALITY</div>
          <div className="text-lg font-bold text-blue-400 font-mono">
            {evaluation.outcome_score !== undefined ? `${Math.round(evaluation.outcome_score * 100)}%` : '--'}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">SAFETY STATUS</div>
          <div className="text-lg font-bold text-slate-100 font-mono">
            {evaluation.safety.has_critical_failure ? 'VIOLATION' : 'CLEAN (0 violations)'}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">VERDICT</div>
          <span className={`inline-block mt-1 text-xs font-bold px-2 py-0.5 rounded ${evaluation.passed ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
            {evaluation.verdict || (evaluation.passed ? 'PASS' : 'FAIL')}
          </span>
        </div>
      </div>
    </div>
  );
};
