/** Safe, explicitly simulated fallback. Never present this as retrieved evidence. */
export function copilotDemo(message: string, patientId: string) {
  if (/suicid|kill myself|hurt myself|end my life/i.test(message)) return "I'm sorry you're going through this. If you might hurt yourself, please contact emergency services or a crisis line now, and ask someone you trust to stay with you. I'm an AI assistant, not a clinician.";
  if (/scared|afraid|terrified|overwhelmed|hopeless/i.test(message)) return "Let's pause and go through this together. I'm an AI assistant, and your oncology team or someone you trust can help support you through this. Would you like help preparing one question for your care team?";
  if (/prescrib|dosage|what dose|which treatment|decide|should i take|stop taking|how long.*live/i.test(message)) return "I'm an AI assistant, not a physician. That's a decision to make together with your oncology team. I can help you prepare questions about the benefits, risks, and uncertainties.";
  return `I'm an AI assistant, not a physician. This is simulated guidance for ${patientId}; no live clinical evidence has been retrieved. I can help organize questions about the available reports for your oncology team, but I cannot choose treatment or verify trial eligibility.`;
}
