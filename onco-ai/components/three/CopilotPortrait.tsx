"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useTexture } from "@react-three/drei";
import * as THREE from "three";
import { useSceneActivity } from "./useSceneActivity";

function Portrait({ active }: { active: boolean }) {
  const texture = useTexture("/doc-ai/doc-ai-cutout.png");
  const model = useRef<THREE.Group>(null);
  useFrame(({ pointer, clock }, delta) => {
    if (!model.current || document.hidden) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    model.current.rotation.y = THREE.MathUtils.damp(model.current.rotation.y, reduced ? 0 : pointer.x * .07, 4, delta);
    model.current.position.y = reduced ? 0 : Math.sin(clock.elapsedTime * (active ? 2 : 1.1)) * .014;
  });
  return <group ref={model}>
    <mesh position={[-.24,-2.15,0]}><planeGeometry args={[4.8,5.92]}/><meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
    {[.77,1.02,1.2].map((radius,i) => <mesh key={radius} position={[0,-.85,-.18-i*.1]} rotation={[1.2,.1,0]}><torusGeometry args={[radius,.008,8,80]}/><meshBasicMaterial color={active ? "#eac473" : "#36d7ff"} transparent opacity={.45-i*.1}/></mesh>)}
    {Array.from({ length: 24 }, (_, i) => <mesh key={i} position={[Math.sin(i*7.4)*1.3,Math.cos(i*3.7)*1.7,-.3]}><sphereGeometry args={[.009,6,6]}/><meshBasicMaterial color="#69dfff"/></mesh>)}
  </group>;
}

/** Layered WebGL portrait using the supplied character, not a rigged 3D human. */
export function CopilotPortrait({ active }: { active: boolean }) {
  const { ref: sceneRef, active: sceneActive } = useSceneActivity<HTMLDivElement>();
  return <div ref={sceneRef} className="copilot-webgl" role="img" aria-label="Doc AI animated illustrative virtual assistant"><Canvas frameloop={sceneActive ? "always" : "demand"} orthographic camera={{ position:[0,0,5], zoom:125 }} dpr={[1,1.5]} gl={{ alpha:true,antialias:true }}><Suspense fallback={null}><Portrait active={active}/></Suspense></Canvas></div>;
}
