import React from 'react';
import { TraceEvent } from '../types';

interface TraceTimelineProps {
  trace: TraceEvent[];
}

export const TraceTimeline: React.FC<TraceTimelineProps> = ({ trace }) => {
  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">3. Live Agent Trace</h2>
      <div className="max-h-72 overflow-y-auto space-y-2.5 pr-2 font-mono text-xs">
        {trace.length === 0 ? (
          <div className="text-slate-600 italic">No events recorded yet.</div>
        ) : (
          trace.map((t, idx) => (
            <div key={idx} className="flex items-start gap-3 py-1 border-b border-slate-900 last:border-0">
              <span className="text-slate-500 shrink-0">{t.timestamp ? t.timestamp.slice(11, 19) : '--:--:--'}</span>
              <span className={t.status === 'warning' ? 'text-amber-400' : 'text-emerald-400'}>
                {t.status === 'warning' ? '⚠' : '✓'}
              </span>
              <span className="text-slate-300 flex-1">{t.summary}</span>
              {t.task_id && <span className="text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded text-[10px]">{t.task_id}</span>}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
