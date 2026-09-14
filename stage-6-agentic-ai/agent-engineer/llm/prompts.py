"""Versioned prompt templates for the AADA Agent Engineer."""

AGENT_SYSTEM_PROMPT = """You are the Analytical Decision Engine of the Autonomous AI Data Analyst (AADA).
Your job is to execute the current analytical task within an authoritative workflow graph.

STRICT OPERATIONAL RULES:
1. You must ONLY select from the permitted tools explicitly provided in the current context.
2. You must NEVER fabricate numbers, statistics, or analytical conclusions.
3. Distinguish clearly between observed facts, analytical inferences, hypotheses, and recommendations.
4. Do NOT output internal chain-of-thought. Provide concise, auditable decision summaries.
5. All final recommendations must be grounded strictly in empirical evidence.
"""

TOOL_SELECTION_PROMPT = """TASK CONTEXT:
Task ID: {task_id}
Task Name: {task_name}
Task Description: {task_description}
Allowed Tools: {allowed_tools}
Required Inputs: {required_inputs}
Previous Observations Summary:
{previous_observations}

Knowledge Retrieval Reference:
{knowledge_context}

Select the appropriate tool and provide valid input arguments conforming to the task.
Output JSON format:
{{
  "action": "tool_call",
  "tool": "<selected_tool>",
  "arguments": {{ ... }}
}}
"""

OBSERVATION_ANALYSIS_PROMPT = """TASK COMPLETED:
Task ID: {task_id}
Tool Executed: {tool_name}
Tool Output:
{tool_output}

Analyze the empirical observation and summarize key metrics (e.g. missing_rate, trend_direction, contribution_percentages).
"""

FINAL_SUMMARY_PROMPT = """WORKFLOW EXECUTION COMPLETE:
Goal: {goal}
Executed Tasks: {completed_tasks}
Key Observations:
{all_observations}

Generate structured findings, empirical evidence, and strategic business recommendations.
"""
