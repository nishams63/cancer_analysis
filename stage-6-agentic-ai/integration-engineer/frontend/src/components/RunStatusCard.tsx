import React from 'react';
import { RunStatus } from '../types';

interface RunStatusCardProps {
  status?: RunStatus;
}

export const RunStatusCard: React.FC<RunStatusCardProps> = ({ status }) => {
  if (!status) return null;

  const getStatusBadge = (s: string) => {
    switch (s) {
      case 'COMPLETED':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'RUNNING':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20 animate-pulse';
      case 'WAITING_FOR_HUMAN':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'FAILED':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">2. Run Status</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <div className="text-xs text-slate-500 font-mono">RUN ID</div>
          <div className="text-sm font-bold text-slate-100 font-mono">{status.run_id}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">STATUS</div>
          <span className={`inline-block mt-0.5 text-xs font-semibold px-2 py-0.5 rounded border ${getStatusBadge(status.status)}`}>
            {status.status}
          </span>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">PROGRESS</div>
          <div className="text-sm font-semibold text-slate-200">
            {status.progress.completed} / {status.progress.total_tasks} tasks ({status.progress.percent}%)
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 font-mono">CONFIDENCE</div>
          <div className="text-sm font-semibold text-emerald-400">
            {status.confidence !== undefined ? `${Math.round(status.confidence * 100)}%` : 'Calculating...'}
          </div>
        </div>
      </div>
    </div>
  );
};
