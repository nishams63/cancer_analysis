const BASE_URL = '/api/v1/aada';

export async function startRun(goal: string): Promise<{ run_id: string; status: string }> {
  const res = await fetch(`${BASE_URL}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal, sync: false }),
  });
  if (!res.ok) throw new Error(`Failed to start run: ${res.statusText}`);
  return res.json();
}

export async function getRunFull(runId: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to fetch run: ${res.statusText}`);
  return res.json();
}

export async function getRunStatus(runId: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}/status`);
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
  return res.json();
}

export async function getRunTrace(runId: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}/trace`);
  if (!res.ok) throw new Error(`Failed to fetch trace: ${res.statusText}`);
  return res.json();
}

export async function approveRun(runId: string, reason: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  return res.json();
}

export async function rejectRun(runId: string, reason: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  return res.json();
}

export async function overrideRun(runId: string, decision: string, reason: string) {
  const res = await fetch(`${BASE_URL}/runs/${runId}/override`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, reason }),
  });
  return res.json();
}
