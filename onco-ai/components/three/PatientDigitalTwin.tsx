"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { AdaptiveDpr, Html, OrbitControls } from "@react-three/drei";
import { gsap } from "gsap";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { ThoracicVolume } from "./ThoracicVolume";
import { MolecularVolume } from "./MolecularVolume";
import { useSceneActivity } from "./useSceneActivity";
import { Activity, Brain, Bone, Expand, Focus, RotateCcw, ScanLine } from "lucide-react";

export type Organ = "Lungs" | "Brain" | "Liver" | "Bone" | "Nodes";
type ViewMode = "3D View" | "CT Scan" | "Pathology" | "Gene Map";

const focus: Record<Organ, { camera: [number, number, number]; target: [number, number, number] }> = {
  Lungs: { camera: [0, .15, 4.6], target: [0, .15, 0] },
  Brain: { camera: [0, .1, 5.7], target: [0, .1, 0] },
  Liver: { camera: [.45, -.15, 4.35], target: [.36, -.18, 0] },
  Bone: { camera: [0, -.6, 5], target: [0, -.45, 0] },
  Nodes: { camera: [-.35, .5, 4], target: [-.32, .52, 0] },
};

const palette: Record<ViewMode, { shell: string; organ: string; line: string; accent: string }> = {
  "3D View": { shell: "#35d6f2", organ: "#2588ff", line: "#35d6f2", accent: "#f2b84b" },
  "CT Scan": { shell: "#d7eff5", organ: "#9db7bf", line: "#f4f7fa", accent: "#ff596a" },
  Pathology: { shell: "#916bff", organ: "#e563ca", line: "#c78cff", accent: "#ffb15b" },
  "Gene Map": { shell: "#27d6b0", organ: "#35d6f2", line: "#39d9a0", accent: "#916bff" },
};

function CameraController({ organ, controls, revision }: { organ: Organ; controls: React.RefObject<OrbitControlsImpl | null>; revision: number }) {
  const { camera, invalidate } = useThree();
  useEffect(() => {
    const next = focus[organ];
    const position = gsap.to(camera.position, { x: next.camera[0], y: next.camera[1], z: next.camera[2], duration: 1.15, ease: "power3.inOut", onUpdate: invalidate });
    const target = controls.current?.target;
    const targetTween = target ? gsap.to(target, { x: next.target[0], y: next.target[1], z: next.target[2], duration: 1.15, ease: "power3.inOut", onUpdate: () => { controls.current?.update(); invalidate(); } }) : null;
    return () => { position.kill(); targetTween?.kill(); };
  }, [camera, controls, invalidate, organ, revision]);
  return null;
}

function PulseTumor({ onSelect }: { onSelect: () => void }) {
  const core = useRef<THREE.Mesh>(null);
  const glow = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (document.hidden) return;
    const pulse = 1 + Math.sin(clock.elapsedTime * 2.1) * .075;
    core.current?.scale.setScalar(pulse);
    glow.current?.scale.setScalar(1.1 + Math.sin(clock.elapsedTime * 1.6) * .1);
  });
  return <group position={[-.39, .67, .32]} onClick={(event) => { event.stopPropagation(); onSelect(); }}>
    <mesh ref={glow}><sphereGeometry args={[.27, 32, 32]} /><meshBasicMaterial color="#ff643c" transparent opacity={.13} depthWrite={false} /></mesh>
    <mesh ref={core}><icosahedronGeometry args={[.14, 4]} /><meshStandardMaterial color="#ff6a3e" emissive="#ff351b" emissiveIntensity={2.1} roughness={.28} /></mesh>
    <Html position={[-.3, .38, 0]} center className="three-label tumor-label"><b>DEMO LESION</b><span>Illustrative location</span></Html>
  </group>;
}

function HolographicPlatform({ color }: { color: string }) {
  const rotating = useRef<THREE.Group>(null);
  useFrame((_, delta) => { if (!document.hidden && rotating.current) rotating.current.rotation.z += delta * .08; });
  return <group position={[0, -1.15, 0]} rotation={[Math.PI / 2, 0, 0]}>
    {[.72, 1, 1.3, 1.58].map((radius, index) => <mesh key={radius}><torusGeometry args={[radius, index === 0 ? .025 : .009, 12, 96]} /><meshBasicMaterial color={color} transparent opacity={.6 - index * .1} /></mesh>)}
    <group ref={rotating}>{[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map((angle) => <mesh key={angle} position={[Math.cos(angle) * 1.45, Math.sin(angle) * 1.45, .01]} rotation={[0, 0, angle]}><boxGeometry args={[.28, .018, .018]} /><meshBasicMaterial color="#f2b84b" /></mesh>)}</group>
  </group>;
}

function SpatialParticles({ color }: { color: string }) {
  const points = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const values = new Float32Array(120 * 3);
    for (let index = 0; index < 120; index += 1) {
      const angle = index * 2.399, radius = 1.45 + (index % 9) * .055;
      values[index * 3] = Math.cos(angle) * radius; values[index * 3 + 1] = ((index % 30) / 30) * 3.6 - 1.7; values[index * 3 + 2] = Math.sin(angle) * radius * .55;
    }
    return values;
  }, []);
  useFrame((_, delta) => { if (!document.hidden && points.current) points.current.rotation.y += delta * .035; });
  return <points ref={points}><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry><pointsMaterial color={color} size={.018} transparent opacity={.52} depthWrite={false} /></points>;
}

function AnatomyModel({ organ, mode, onSelect }: { organ: Organ; mode: ViewMode; onSelect: (organ: Organ) => void }) {
  const colors = palette[mode];
  if (mode === "Gene Map" || mode === "Pathology") return <group><MolecularVolume pathology={mode === "Pathology"}/><HolographicPlatform color={colors.line}/></group>;
  return <group position={[0, -.1, 0]}>
    <ThoracicVolume color={colors.line} ct={mode === "CT Scan"} onSelect={() => onSelect("Lungs")} />
    <PulseTumor onSelect={() => onSelect("Lungs")} />
    <group onClick={() => onSelect("Nodes")}>{[[.58,.43,.25],[.48,.22,.3],[.62,.05,.2]].map((position,index) => <mesh key={index} position={position as [number,number,number]}><sphereGeometry args={[.065,24,24]} /><meshStandardMaterial color="#916bff" emissive="#7655ff" emissiveIntensity={organ === "Nodes" ? 2 : .85} /></mesh>)}<Html position={[.9,.65,.12]} center className="three-label node-label"><b>LYMPH NODES</b><span>Demo markers</span></Html></group>
    <Html position={[.8,-.65,.08]} center className="three-label risk-label"><b>ILLUSTRATIVE MODEL</b><span>Not patient imaging</span></Html>
    <HolographicPlatform color={colors.line} /><SpatialParticles color={colors.line} />
  </group>;
}

function LoadingTwin() { return <div className="twin-loading"><span /><b>Initializing spatial patient model</b></div>; }

export function PatientDigitalTwin({ patientId }: { patientId: string }) {
  const { ref: sceneRef, active: sceneActive } = useSceneActivity<HTMLElement>();
  const [revision, setRevision] = useState(0);
  const [organ, setOrgan] = useState<Organ>("Lungs"), [mode, setMode] = useState<ViewMode>("3D View");
  const selectOrgan = (next: Organ) => { setOrgan(next); setRevision(value => value + 1); };
  const controls = useRef<OrbitControlsImpl>(null);
  const reset = () => { selectOrgan("Lungs"); controls.current?.reset(); };
  const fullScreen = () => { if (document.fullscreenElement) void document.exitFullscreen(); else void document.querySelector(".digital-twin-card")?.requestFullscreen?.(); };
  const organIcons: Record<Organ, typeof ScanLine> = { Lungs: ScanLine, Brain, Liver: Activity, Bone, Nodes: Focus };
  return <section ref={sceneRef} className="card digital-twin-card">
    <div className="twin-toolbar"><div><p className="eyebrow">3D DIGITAL TWIN</p><h2>Spatial disease intelligence</h2></div><div className="twin-actions"><span className="live-signal"><i /> WEBGL · {patientId}</span><button onClick={reset} aria-label="Reset digital twin view"><RotateCcw /></button><button onClick={fullScreen} aria-label="Open digital twin fullscreen"><Expand /></button></div></div>
    <div className="twin-modes" role="tablist" aria-label="Visualization modes">{(["3D View","CT Scan","Pathology","Gene Map"] as ViewMode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} onClick={() => { setMode(item); setOrgan("Lungs"); controls.current?.reset(); }} role="tab" aria-selected={mode === item}>{item}</button>)}</div>
    <div className="twin-stage"><Suspense fallback={<LoadingTwin />}><Canvas frameloop={sceneActive ? "always" : "demand"} camera={{ position: [0,.1,5.7], fov: 38, near: .1, far: 100 }} dpr={[1,1.5]} gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}>
      <color attach="background" args={["#020a11"]} /><fog attach="fog" args={["#020a11", 5.5, 10]} /><ambientLight intensity={.7} /><pointLight color="#35d6f2" position={[2,3,4]} intensity={18} distance={9} /><pointLight color="#f2b84b" position={[-3,1,2]} intensity={10} distance={8} />
      <AnatomyModel organ={organ} mode={mode} onSelect={selectOrgan} /><OrbitControls ref={controls} makeDefault enablePan={false} enableDamping dampingFactor={.075} minDistance={3.3} maxDistance={7} minPolarAngle={Math.PI * .26} maxPolarAngle={Math.PI * .72} /><CameraController organ={organ} controls={controls} revision={revision} /><AdaptiveDpr pixelated />
    </Canvas></Suspense>
      <div className="organ-selector vertical" aria-label="Choose organ focus">{(Object.keys(organIcons) as Organ[]).map((item) => { const Icon = organIcons[item] as typeof ScanLine; return <button key={item} className={organ === item ? "active" : ""} onClick={() => selectOrgan(item)}><Icon /><span>{item}</span></button>; })}</div>
      <div className="twin-mode-readout"><span>{mode} · illustrative demo</span><b>{organ === "Brain" || organ === "Liver" ? `No ${organ.toLowerCase()} imaging linked` : `${organ} focus`}</b></div><small className="twin-hint">Drag to rotate · Scroll to zoom · Not diagnostic imaging</small>
    </div>
    <div className="twin-footer-controls">{(["3D View","CT Scan","Pathology","Gene Map"] as ViewMode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)}>{item}</button>)}<button onClick={reset}><RotateCcw /> Reset View</button></div>
  </section>;
}
