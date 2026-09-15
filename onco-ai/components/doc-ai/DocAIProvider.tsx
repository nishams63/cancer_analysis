"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type DocEmotion = "idle" | "thinking" | "speaking" | "attention" | "warning";
type SpeechRecognitionResultEvent = { results: { [index: number]: { [index: number]: { transcript: string } } } };
type SpeechRecognitionInstance = {
  lang: string; interimResults: boolean; continuous: boolean;
  start: () => void; stop: () => void;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  onerror: (() => void) | null; onend: (() => void) | null;
};
type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

type DocAIContextValue = {
  open: boolean; emotion: DocEmotion; message: string; suggestions: string[];
  voiceEnabled: boolean; listening: boolean;
  speak: (message: string) => void; suggest: (actions: string[]) => void;
  setEmotion: (state: DocEmotion) => void;
  startListening: (onTranscript: (text: string) => void) => void;
  toggleVoice: () => void; openPanel: () => void; close: () => void; toggle: () => void;
};

const Context = createContext<DocAIContextValue | null>(null);

export function DocAIProvider({ children, initialMessage }: { children: React.ReactNode; initialMessage: string }) {
  const [open, setOpen] = useState(false);
  const [emotion, setEmotion] = useState<DocEmotion>("idle");
  const [message, setMessage] = useState(initialMessage);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [listening, setListening] = useState(false);

  useEffect(() => () => window.speechSynthesis?.cancel(), []);
  const speak = useCallback((next: string) => {
    setMessage(next); setOpen(true); setEmotion("speaking");
    if (!voiceEnabled || !("speechSynthesis" in window)) {
      window.setTimeout(() => setEmotion("idle"), 900); return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(next);
    utterance.lang = "en-US"; utterance.rate = 0.9; utterance.pitch = 1;
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find((voice) => /^en/i.test(voice.lang) && /rishi|daniel|male/i.test(voice.name)) || voices.find((voice) => /^en/i.test(voice.lang)) || null;
    utterance.onend = () => setEmotion("idle"); utterance.onerror = () => setEmotion("idle");
    window.speechSynthesis.speak(utterance);
  }, [voiceEnabled]);

  const startListening = useCallback((onTranscript: (text: string) => void) => {
    const speechWindow = window as unknown as { SpeechRecognition?: SpeechRecognitionConstructor; webkitSpeechRecognition?: SpeechRecognitionConstructor };
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;
    if (!Recognition) { setMessage("Voice input is not supported in this browser. You can type your question instead."); setOpen(true); return; }
    const recognition = new Recognition();
    recognition.lang = "en-US"; recognition.interimResults = false; recognition.continuous = false;
    recognition.onresult = (event) => onTranscript(event.results[0][0].transcript);
    recognition.onerror = () => { setListening(false); setEmotion("idle"); };
    recognition.onend = () => { setListening(false); setEmotion("idle"); };
    setListening(true); setEmotion("attention"); recognition.start();
  }, []);

  const close = useCallback(() => { window.speechSynthesis?.cancel(); setEmotion("idle"); setOpen(false); }, []);
  const value = useMemo(() => ({
    open, emotion, message, suggestions, voiceEnabled, listening, speak,
    suggest: setSuggestions, setEmotion, startListening,
    toggleVoice: () => setVoiceEnabled((enabled) => { if (enabled) window.speechSynthesis?.cancel(); return !enabled; }),
    openPanel: () => setOpen(true), close, toggle: () => setOpen((value) => !value),
  }), [open, emotion, message, suggestions, voiceEnabled, listening, speak, startListening, close]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useDocAI() {
  const value = useContext(Context);
  if (!value) throw new Error("useDocAI must be used inside DocAIProvider");
  return value;
}
