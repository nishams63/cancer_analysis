import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { GoalInput } from './components/GoalInput';
import { RunStatusCard } from './components/RunStatusCard';
import { TraceTimeline } from './components/TraceTimeline';
import { EvidencePanel } from './components/EvidencePanel';
import { EvaluationCard } from './components/EvaluationCard';
import { HumanReviewModal } from './components/HumanReviewModal';
import { ResultPanel } from './components/ResultPanel';
import { startRun, getRunFull, getRunTrace, approveRun, rejectRun, overrideRun } from './api/client';

export default function App() {
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [runData, setRunData] = useState<any | null>(null);
  const [trace, setTrace] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const handleStartRun = async (goal: string) => {
    setLoading(true);
    try {
      const res = await startRun(goal);
      setCurrentRunId(res.run_id);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!currentRunId) return;

    const interval = setInterval(async () => {
      try {
        const full = await getRunFull(currentRunId);
        setRunData(full);
        const tr = await getRunTrace(currentRunId);
        setTrace(tr);

        if (['COMPLETED', 'FAILED', 'CANCELLED'].includes(full.status)) {
          clearInterval(interval);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [currentRunId]);

  const handleApprove = async (reason: string) => {
    if (!currentRunId) return;
    await approveRun(currentRunId, reason);
  };

  const handleReject = async (reason: string) => {
    if (!currentRunId) return;
    await rejectRun(currentRunId, reason);
  };

  const handleOverride = async (decision: string, reason: string) => {
    if (!currentRunId) return;
    await overrideRun(currentRunId, decision, reason);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <Header />
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        <GoalInput onStart={handleStartRun} loading={loading} />

        {runData && (
          <>
            <HumanReviewModal
              review={runData.human_review}
              onApprove={handleApprove}
              onReject={handleReject}
              onOverride={handleOverride}
            />

            <RunStatusCard status={runData} />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TraceTimeline trace={trace} />
              <EvidencePanel analysis={runData.analysis} />
            </div>

            <EvaluationCard evaluation={runData.evaluation} />
            <ResultPanel analysis={runData.analysis} />
          </>
        )}
      </main>
    </div>
  );
}
