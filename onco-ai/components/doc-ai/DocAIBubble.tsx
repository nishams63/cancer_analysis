"use client";

import { motion } from "framer-motion";
import { Sparkles, Volume2 } from "lucide-react";
import { useDocAI } from "./DocAIProvider";

export function DocAIBubble() {
  const { message, emotion, voiceEnabled } = useDocAI();
  return <motion.div className={`doc-bubble is-${emotion}`} initial={{ opacity: 0, scale: .96, y: 6 }} animate={{ opacity: 1, scale: emotion === "speaking" ? 1.01 : 1, y: 0 }} transition={{ duration: .24 }}>
    <div className="doc-bubble-title"><Sparkles /> Doc AI <span>Virtual oncology assistant</span>{voiceEnabled && <Volume2 className="bubble-voice" />}</div>
    <p>{message}</p>
    {emotion === "thinking" && <div className="typing" aria-label="Doc AI is thinking"><i /><i /><i /></div>}
  </motion.div>;
}
