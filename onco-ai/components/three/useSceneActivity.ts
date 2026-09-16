"use client";
import { useEffect, useRef, useState } from "react";

export function useSceneActivity<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [active, setActive] = useState(true);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    let visible = true;
    const update = () => setActive(visible && !document.hidden && !media.matches);
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; update(); });
    if (ref.current) observer.observe(ref.current);
    document.addEventListener("visibilitychange", update);
    media.addEventListener("change", update);
    update();
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", update); media.removeEventListener("change", update); };
  }, []);
  return { ref, active };
}
