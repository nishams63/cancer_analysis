"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Mic, MicOff, Send, ShieldCheck, Volume2, VolumeX, X } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import { useDocAI } from "./DocAIProvider";

function createSafeResponse(question: string) {
  const normalized = question.toLowerCase();
  if (/best treatment|choose.*treatment|final decision|what should i take|recommend.*treatment/.test(normalized)) return "That’s a decision to make together with your oncology team. I can help you prepare questions for that conversation.";
  if (/afraid|scared|terrified|panic|hopeless|can.?t cope|crisis/.test(normalized)) return "Let’s pause for a moment. You don’t have to process this alone; please consider contacting your care team or someone you trust, and I can stay with you while we organize one question at a time.";
  if (/report|scan|lab|result/.test(normalized)) return "I can help explain the report in plain language, but I need the actual report or result to avoid guessing. Which report would you like to go through together?";
  if (/prognosis|how long|months to live|survival/.test(normalized)) return "Prognosis depends on details that only your oncology team can interpret in context, so I won’t estimate an outcome. I can help you write a clear question to ask them about what the available information means for you.";
  return "This demo only contains synthetic information for ONC-2048, so I won’t invent missing clinical facts. I can explain a term or help turn your concern into a question to discuss with your oncologist.";
}

export function DocAIPanel() {
  const ai = useDocAI();
  const [input, setInput] = useState("");
  function answer(question: string) { if (!question.trim()) return; ai.setEmotion("thinking"); setInput(""); window.setTimeout(() => ai.speak(createSafeResponse(question)), 550); }

  return <AnimatePresence>{ai.open && <motion.aside className={`doc-panel state-${ai.emotion}`} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 24 }} transition={{ duration: .22 }} aria-label="Doc AI virtual oncology information assistant">
    <header>
      <div className="assistant-avatar"><Image src="/doc-ai/doc-ai.png" alt="" width={1129} height={1393} /></div>
      <div><b>Doc AI</b><span>Virtual oncology information assistant</span></div>
      <span className="not-physician">Not a physician</span>
      <button onClick={ai.close} aria-label="Close Doc AI"><X /></button>
    </header>
    <div className="assistant-thread">
      <div className="panel-character" aria-hidden="true"><Image src="/doc-ai/doc-ai.png" alt="" width={1129} height={1393} /></div>
      <div className="assistant-response">
        <div className="assistant-message">{ai.message}</div>
        {ai.emotion === "thinking" && <div className="typing panel-typing" aria-label="Doc AI is thinking"><i /><i /><i /></div>}
        <div className="safety-note"><ShieldCheck /> I’m an AI assistant, not a licensed physician. Treatment information is for discussion with your oncology team.</div>
      </div>
    </div>
    <div className="assistant-suggestions">{(ai.suggestions.length ? ai.suggestions : ["Explain this in plain language", "Help me prepare a doctor question", "Which report should we review?"]).map((suggestion) => <button key={suggestion} onClick={() => answer(suggestion)}>{suggestion}</button>)}</div>
    <form onSubmit={(event) => { event.preventDefault(); answer(input); }}>
      <button type="button" onClick={ai.toggleVoice} aria-label={ai.voiceEnabled ? "Mute Doc AI voice" : "Enable Doc AI voice"} title="Toggle spoken responses">{ai.voiceEnabled ? <Volume2 /> : <VolumeX />}</button>
      <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about an oncology term or report…" aria-label="Message Doc AI" />
      <button type="button" className={ai.listening ? "listening" : ""} onClick={() => ai.startListening((text) => { setInput(text); answer(text); })} aria-label={ai.listening ? "Listening for your question" : "Ask with your voice"}>{ai.listening ? <MicOff /> : <Mic />}</button>
      <button className="send" aria-label="Send message"><Send /></button>
    </form>
    <footer>Voice responses are short, calm, and informational. No diagnosis or prescription is provided.</footer>
  </motion.aside>}</AnimatePresence>;
}
