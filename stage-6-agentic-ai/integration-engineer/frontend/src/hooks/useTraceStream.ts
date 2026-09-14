import { useState, useEffect } from 'react';

export function useTraceStream(runId: string | null) {
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    if (!runId) return;

    const eventSource = new EventSource(`/api/v1/aada/runs/${runId}/stream`);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setEvents((prev) => [...prev, parsed]);
      } catch (e) {
        // Ping or parse error
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [runId]);

  return events;
}
