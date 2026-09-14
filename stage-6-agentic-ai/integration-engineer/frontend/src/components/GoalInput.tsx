import React, { useState } from 'react';

interface GoalInputProps {
  onStart: (goal: string) => void;
  loading: boolean;
}

export const GoalInput: React.FC<GoalInputProps> = ({ onStart, loading }) => {
  const [goal, setGoal] = useState('Analyze why revenue decreased last quarter.');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (goal.trim() && !loading) {
      onStart(goal);
    }
  };

  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 shadow-sm">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">1. Goal Input</h2>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="e.g. Analyze why revenue decreased last quarter"
          className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !goal.trim()}
          className="bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-slate-950 font-semibold text-sm px-6 py-2.5 rounded-lg transition-colors flex items-center gap-2"
        >
          {loading ? 'Running Analysis...' : 'Run Analysis'}
        </button>
      </form>
    </div>
  );
};
