import React, { useState } from 'react';
import { HumanReviewInfo } from '../types';

interface HumanReviewModalProps {
  review?: HumanReviewInfo;
  onApprove: (reason: string) => void;
  onReject: (reason: string) => void;
  onOverride: (decision: string, reason: string) => void;
}

export const HumanReviewModal: React.FC<HumanReviewModalProps> = ({
  review,
  onApprove,
  onReject,
  onOverride,
}) => {
  const [reason, setReason] = useState('Reviewed by analyst.');
  const [overrideDecision, setOverrideDecision] = useState('Select Hypothesis A');

  if (!review || !review.required || review.status !== 'PENDING') return null;

  return (
    <div className="bg-amber-950/40 border-2 border-amber-500/50 rounded-xl p-6 shadow-lg">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-amber-400 text-lg">⏸</span>
        <h2 className="text-base font-bold text-amber-300 uppercase tracking-wide">Human Review Required</h2>
      </div>
      <p className="text-sm text-slate-200 mb-2">
        <strong className="text-slate-400">Reason:</strong> {review.reason || 'Confidence margin requires analyst verification.'}
      </p>
      {review.current_task && (
        <p className="text-xs text-slate-400 mb-4 font-mono">Current Task: {review.current_task}</p>
      )}

      <div className="space-y-3 mb-4">
        <div>
          <label className="block text-xs font-semibold text-slate-400 mb-1">Analyst Rationale / Override</label>
          <input
            type="text"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-xs text-slate-100"
          />
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => onApprove(reason)}
          className="bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs px-4 py-2 rounded transition-colors"
        >
          APPROVE
        </button>
        <button
          onClick={() => onReject(reason)}
          className="bg-rose-500 hover:bg-rose-600 text-white font-bold text-xs px-4 py-2 rounded transition-colors"
        >
          REJECT
        </button>
        <button
          onClick={() => onOverride(overrideDecision, reason)}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold text-xs px-4 py-2 rounded transition-colors"
        >
          OVERRIDE
        </button>
      </div>
    </div>
  );
};
