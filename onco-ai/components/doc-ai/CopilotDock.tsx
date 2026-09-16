"use client";

import dynamic from "next/dynamic";
import { FormEvent, PointerEvent, useRef, useState } from "react";
import { BrainCircuit, FileText, FlaskConical, GitCompareArrows, Mic, Send, ShieldCheck, Sparkles, TestTube2, Volume2, VolumeX, X } from "lucide-react";
import { usePatientStore } from "@/stores/patient-store";
import { useDocAI } from "./DocAIProvider";

const CopilotPortrait = dynamic(() => import("@/components/three/CopilotPortrait").then(module => module.CopilotPortrait), { ssr: false });

const actions = [
  ["Summarize this patient", FileText], ["Explain toxicity risk", FlaskConical], ["Explain progression", TestTube2],
  ["Find clinical trials", Sparkles], ["Compare treatment options", GitCompareArrows], ["Generate physician handoff note", BrainCircuit],
] as const;

export function CopilotDock() {
  const { activePatient } = usePatientStore();
  const ai = useDocAI();
  const scene = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");

  function parallax(event: PointerEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width - .5) * 2;
    const y = ((event.clientY - box.top) / box.height - .5) * 2;
    scene.current?.style.setProperty("--avatar-x", `${x * 8}px`);
    scene.current?.style.setProperty("--avatar-y", `${y * 5}px`);
    scene.current?.style.setProperty("--ring-x", `${x * -5}px`);
  }

  async function ask(question: string) {
    if (!question.trim()) return;
    setInput(""); setError(""); ai.setEmotion("thinking");
    try {
      const response = await fetch("/api/slm/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patientId: activePatient.id, inputVersion: "command-center-v2", message: question }) });
      const payload = await response.json() as { response?: string; error?: string };
      if (!response.ok) throw new Error(payload.error || "Copilot service unavailable");
      ai.speak(payload.response || `I reviewed the available synthetic context for ${activePatient.id}. The evidence is ready for physician review.`);
    } catch {
      setError("The clinical service is unavailable. Demo-safe guidance is shown instead.");
      ai.speak(`I can still organize the synthetic dashboard signals for ${activePatient.id}, but live evidence retrieval is unavailable. Physician review is required.`);
    }
  }

  function submit(event: FormEvent) { event.preventDefault(); void ask(input); }

  return <><button className="copilot-toggle" onClick={ai.toggle}><BrainCircuit size={20}/>Clinical Copilot</button><aside className={`copilot-dock copilot-${ai.emotion} ${ai.open ? "is-open" : ""}`} aria-label="ONCO.AI Clinical Copilot">
    <header><span className="copilot-logo"><BrainCircuit /></span><div><b>ONCO.AI</b><small>Clinical Copilot</small></div><i /><em>Demo</em><button className="copilot-close" onClick={ai.close} aria-label="Close copilot"><X/></button></header>
    <div className="copilot-scene" ref={scene} onPointerMove={parallax} onPointerLeave={() => { scene.current?.style.setProperty("--avatar-x", "0px"); scene.current?.style.setProperty("--avatar-y", "0px"); }}>
      <div className="holo-ring ring-a" /><div className="holo-ring ring-b" /><div className="holo-ring ring-c" />
      <div className="copilot-particles">{Array.from({ length: 14 }, (_, index) => <i key={index} style={{ left: `${8 + (index * 17) % 84}%`, top: `${10 + (index * 23) % 72}%`, animationDelay: `${index * -.31}s` }} />)}</div>
      <CopilotPortrait active={ai.emotion === "speaking" || ai.listening} />
      <div className="holo-lung-panel"><span /><span /><b>Thoracic signal map</b></div>
      <div className="copilot-greeting"><b>Good morning, Dr. Sharma.</b><span>I&apos;ve reviewed the latest signals for {activePatient.id}.</span></div>
      <div className="copilot-risk-stack"><span className="risk-red">Toxicity <b>↑</b></span><span className="risk-violet">Progression <b>↑</b></span><span className="risk-gold">Urgency <b>{activePatient.status === "Stable" ? "ROUTINE" : "HIGH"}</b></span></div>
      <div className="copilot-platform"><i /><i /><i /></div>
    </div>
    <section className="copilot-conversation">
      <div className="copilot-message"><Sparkles /><p>{ai.message}</p></div>
      <h3>How can I help you today?</h3>
      <div className="copilot-actions">{actions.map(([label, Icon]) => <button key={label} onClick={() => void ask(label)}><Icon /><span>{label}</span></button>)}</div>
      <form onSubmit={submit}><input value={input} onChange={(event) => setInput(event.target.value)} placeholder={`Ask about ${activePatient.id}...`} aria-label={`Ask about ${activePatient.id}`} /><button type="button" className={ai.listening ? "listening" : ""} onClick={() => ai.startListening((text) => void ask(text))} aria-label="Ask with microphone"><Mic /></button><button className="send" aria-label="Send to Clinical Copilot"><Send /></button></form>
      {error && <p className="copilot-error">{error}</p>}
      <div className="copilot-state"><span className="waveform"><i/><i/><i/><i/><i/></span><b>{ai.emotion === "thinking" ? "Retrieving evidence…" : ai.emotion === "speaking" ? "Speaking…" : ai.listening ? "Listening…" : "Ready"}</b><button className="copilot-voice" onClick={ai.toggleVoice} aria-label={ai.voiceEnabled ? "Mute voice" : "Enable voice"}>{ai.voiceEnabled ? <Volume2/> : <VolumeX/>}</button></div>
      <div className="copilot-safety"><ShieldCheck /> AI assistant, not a physician. Discuss treatment information with your oncology team.</div>
    </section>
  </aside></>;
}
