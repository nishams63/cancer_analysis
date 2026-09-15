"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Mic, MicOff, Send, ShieldCheck, Volume2, VolumeX, X } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import { useDocAI } from "./DocAIProvider";
import { usePatientStore } from "@/stores/patient-store";

function createSafeResponse(question: string, patientId: string) {
  const normalized = question.toLowerCase();
  if (/best treatment|choose.*treatment|final decision|what should i take|recommend.*treatment/.test(normalized)) return "That’s a decision to make together with your oncology team. I can help you prepare questions for that conversation.";
  if (/afraid|scared|terrified|panic|hopeless|can.?t cope|crisis/.test(normalized)) return "Let’s pause for a moment. You don’t have to process this alone; please consider contacting your care team or someone you trust, and I can stay with you while we organize one question at a time.";
  if (/report|scan|lab|result/.test(normalized)) return "I can help explain the report in plain language, but I need the actual report or result to avoid guessing. Which report would you like to go through together?";
  if (/prognosis|how long|months to live|survival/.test(normalized)) return "Prognosis depends on details that only your oncology team can interpret in context, so I won’t estimate an outcome. I can help you write a clear question to ask them about what the available information means for you.";
  if (/summarize|patient/.test(normalized)) return `I’m an AI assistant, not a physician. ${patientId} is a synthetic case currently selected in the dashboard; I can organize its model signals and evidence for physician review, but I won’t make a diagnosis or treatment decision.`;
  if (/toxicity/.test(normalized)) return `Let’s go through this together. The dashboard flags an elevated synthetic toxicity signal for ${patientId}; the contributing factors and confidence should be reviewed with the oncology team before any decision.`;
  if (/progression/.test(normalized)) return `The demo shows increasing longitudinal signals for ${patientId}, but these are not a diagnosis. I can help you list the imaging and laboratory findings to discuss with the oncologist.`;
  if (/trial/.test(normalized)) return `The trial matches are informational candidates only. Eligibility for ${patientId} needs confirmation by the treating oncologist and the study team.`;
  return `I’m reviewing the synthetic record for ${patientId}, and I won’t invent missing clinical facts. I can explain a term or help turn your concern into a question to discuss with your oncologist.`;
}

export function DocAIPanel() {
  const ai = useDocAI();
  const { activePatient } = usePatientStore();
  const [input, setInput] = useState("");
  function answer(question: string) { if (!question.trim()) return; ai.setEmotion("thinking"); setInput(""); window.setTimeout(() => ai.speak(createSafeResponse(question, activePatient.id)), 550); }

  return <AnimatePresence>{ai.open && <motion.aside className={`doc-panel state-${ai.emotion}`} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 24 }} transition={{ duration: .22 }} aria-label="Doc AI virtual oncology information assistant">
    <header>
      <div className="assistant-avatar"><Image src="/doc-ai/doc-ai-cutout.png" alt="" width={1129} height={1393} unoptimized /></div>
      <div><b>ONCO.AI Clinical Copilot</b><span>Reviewing {activePatient.id} · What would you like to investigate?</span></div>
      <span className="not-physician">Not a physician</span>
      <button onClick={ai.close} aria-label="Close Doc AI"><X /></button>
    </header>
    <div className="assistant-thread">
      <div className="panel-character" aria-hidden="true"><Image src="/doc-ai/doc-ai-cutout.png" alt="" width={1129} height={1393} unoptimized /></div>
      <div className="assistant-response">
        <div className="assistant-message">{ai.message}</div>
        {ai.emotion === "thinking" && <div className="typing panel-typing" aria-label="Doc AI is thinking"><i /><i /><i /></div>}
        <div className="safety-note"><ShieldCheck /> I’m an AI assistant, not a licensed physician. Treatment information is for discussion with your oncology team.</div>
      </div>
    </div>
    <div className="assistant-suggestions">{(ai.suggestions.length ? ai.suggestions : ["Summarize patient", "Explain toxicity", "Explain progression", "Find trials", "Compare treatments", "Generate report"]).map((suggestion) => <button key={suggestion} onClick={() => answer(suggestion)}>{suggestion}</button>)}</div>
    <form onSubmit={(event) => { event.preventDefault(); answer(input); }}>
      <button type="button" onClick={ai.toggleVoice} aria-label={ai.voiceEnabled ? "Mute Doc AI voice" : "Enable Doc AI voice"} title="Toggle spoken responses">{ai.voiceEnabled ? <Volume2 /> : <VolumeX />}</button>
      <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about an oncology term or report…" aria-label="Message Doc AI" />
      <button type="button" className={ai.listening ? "listening" : ""} onClick={() => ai.startListening((text) => { setInput(text); answer(text); })} aria-label={ai.listening ? "Listening for your question" : "Ask with your voice"}>{ai.listening ? <MicOff /> : <Mic />}</button>
      <button className="send" aria-label="Send message"><Send /></button>
    </form>
    <footer>Voice responses are short, calm, and informational. No diagnosis or prescription is provided.</footer>
  </motion.aside>}</AnimatePresence>;
}
