export interface RunStatus {
  run_id: string;
  status: 'CREATED' | 'PLANNING' | 'READY' | 'RUNNING' | 'WAITING_FOR_HUMAN' | 'EVALUATING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  current_task?: string;
  completed_tasks: string[];
  failed_tasks: string[];
  confidence?: number;
  escalation_status: string;
  progress: {
    total_tasks: number;
    completed: number;
    percent: number;
  };
}

export interface TraceEvent {
  trace_id: string;
  timestamp: string;
  task_id?: string;
  event_type: string;
  tool_name: string;
  summary: string;
  status: string;
}

export interface AnalysisResult {
  summary: string;
  findings: Array<{ cause: string; score?: number; details?: string }>;
  evidence: Array<{ data: any; classification: 'OBSERVED FACT' | 'EVIDENCE-SUPPORTED INFERENCE' }>;
  recommendations: string[];
  confidence?: number;
  limitations?: string;
}

export interface EvaluationData {
  evaluated: boolean;
  passed: boolean;
  verdict: string;
  process_score?: number;
  outcome_score?: number;
  overall_score?: number;
  process_metrics: Record<string, number>;
  outcome_metrics: Record<string, number>;
  safety: {
    critical_violations: number;
    high_violations: number;
    has_critical_failure: boolean;
  };
  failures: Array<{
    failure_id: string;
    category: string;
    severity: string;
    explanation: string;
    fix: string;
  }>;
}

export interface HumanReviewInfo {
  required: boolean;
  status?: string;
  reason?: string;
  current_task?: string;
  options: string[];
}
