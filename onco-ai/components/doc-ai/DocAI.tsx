"use client";

import { motion } from "framer-motion";
import { MessageCircle } from "lucide-react";
import Image from "next/image";
import { DocAIBubble } from "./DocAIBubble";
import { DocAIPanel } from "./DocAIPanel";
import { useDocAI } from "./DocAIProvider";

export function DocAI() {
  const ai = useDocAI();
  const animation = ai.emotion === "speaking" ? { y: [0, -2, 0], scale: [1, 1.018, 1] } : ai.emotion === "attention" ? { y: [0, -4, 0], scale: [1, 1.01, 1] } : { y: [0, -3, 0], scale: 1 };
  return <>
    <div className={`doc-ai-dock state-${ai.emotion}`}>
      <DocAIBubble />
      <motion.button className="doc-avatar" onClick={ai.toggle} animate={animation} transition={{ repeat: Infinity, duration: ai.emotion === "speaking" ? 1.7 : 4, ease: "easeInOut" }} aria-label="Open Doc AI virtual oncology assistant">
        <span className="doc-image-slot"><Image src="/doc-ai/doc-ai-cutout.png" alt="Doc AI virtual clinical assistant" width={1129} height={1393} priority /></span>
        <MessageCircle className="doc-status" />
      </motion.button>
    </div>
    <DocAIPanel />
  </>;
}
