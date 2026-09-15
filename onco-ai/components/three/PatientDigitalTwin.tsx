"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { AdaptiveDpr, Html, Line, OrbitControls } from "@react-three/drei";
import { gsap } from "gsap";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Activity, Brain, Bone, Expand, Focus, RotateCcw, ScanLine } from "lucide-react";

export type Organ = "Lungs" | "Brain" | "Liver" | "Bone" | "Nodes";
type ViewMode = "3D View" | "CT Scan" | "Pathology" | "Gene Map";

const focus: Record<Organ, { camera: [number, number, number]; target: [number, number, number] }> = {
  Lungs: { camera: [0, .55, 4.3], target: [0, .55, 0] },
  Brain: { camera: [0, 1.9, 4.5], target: [0, 1.82, 0] },
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

function CameraController({ organ, controls }: { organ: Organ; controls: React.RefObject<OrbitControlsImpl | null> }) {
  const { camera, invalidate } = useThree();
  useEffect(() => {
    const next = focus[organ];
    const position = gsap.to(camera.position, { x: next.camera[0], y: next.camera[1], z: next.camera[2], duration: 1.15, ease: "power3.inOut", onUpdate: invalidate });
    const target = controls.current?.target;
    const targetTween = target ? gsap.to(target, { x: next.target[0], y: next.target[1], z: next.target[2], duration: 1.15, ease: "power3.inOut", onUpdate: () => { controls.current?.update(); invalidate(); } }) : null;
    return () => { position.kill(); targetTween?.kill(); };
  }, [camera, controls, invalidate, organ]);
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
    <Html position={[-.2, .24, 0]} center transform distanceFactor={4.4} className="three-label tumor-label"><b>PRIMARY TUMOR</b><span>Detected · 3.2 cm</span></Html>
  </group>;
}

function Lung({ side, color, selected, onSelect }: { side: -1 | 1; color: string; selected: boolean; onSelect: () => void }) {
  const [hovered, setHovered] = useState(false);
  const mesh = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => { if (mesh.current && !document.hidden) mesh.current.scale.y = 1.25 + Math.sin(clock.elapsedTime * 1.15) * .018; });
  return <mesh ref={mesh} position={[side * .32, .47, .02]} scale={[.52, 1.25, .38]} rotation={[0, 0, side * -.1]}
    onPointerOver={(event: ThreeEvent<PointerEvent>) => { event.stopPropagation(); setHovered(true); }} onPointerOut={() => setHovered(false)}
    onClick={(event) => { event.stopPropagation(); onSelect(); }} onDoubleClick={(event) => { event.stopPropagation(); onSelect(); }}>
    <sphereGeometry args={[.54, 48, 48]} />
    <meshPhysicalMaterial color={color} transparent opacity={selected ? .42 : .27} roughness={.15} metalness={.05} transmission={.2} emissive={color} emissiveIntensity={hovered || selected ? .75 : .28} wireframe={hovered} depthWrite={false} />
  </mesh>;
}

function HolographicPlatform({ color }: { color: string }) {
  const rotating = useRef<THREE.Group>(null);
  useFrame((_, delta) => { if (!document.hidden && rotating.current) rotating.current.rotation.z += delta * .08; });
  return <group position={[0, -1.73, 0]} rotation={[Math.PI / 2, 0, 0]}>
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
  const torso = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(-.7, 1.25); shape.bezierCurveTo(-1.12, .92, -1.02, -.25, -.72, -1.4); shape.bezierCurveTo(-.38, -1.72, .38, -1.72, .72, -1.4); shape.bezierCurveTo(1.02, -.25, 1.12, .92, .7, 1.25); shape.bezierCurveTo(.42, 1.46, -.42, 1.46, -.7, 1.25);
    return new THREE.ExtrudeGeometry(shape, { depth: .45, bevelEnabled: true, bevelSize: .07, bevelThickness: .08, bevelSegments: 4, curveSegments: 32 });
  }, []);
  return <group position={[0, -.1, 0]}>
    <mesh geometry={torso} position={[0, 0, -.28]}><meshPhysicalMaterial color={colors.shell} transparent opacity={mode === "CT Scan" ? .11 : .075} roughness={.12} metalness={.15} transmission={.56} thickness={.35} side={THREE.DoubleSide} depthWrite={false} /></mesh>
    <mesh position={[0, 1.68, -.05]} onClick={() => onSelect("Brain")}><sphereGeometry args={[.42, 40, 40]} /><meshPhysicalMaterial color={organ === "Brain" ? colors.accent : colors.shell} transparent opacity={organ === "Brain" ? .35 : .12} emissive={colors.shell} emissiveIntensity={.35} depthWrite={false} /></mesh>
    <Lung side={-1} color={colors.organ} selected={organ === "Lungs"} onSelect={() => onSelect("Lungs")} /><Lung side={1} color={colors.organ} selected={organ === "Lungs"} onSelect={() => onSelect("Lungs")} />
    <mesh position={[.3, -.32, .03]} rotation={[.12, 0, -.16]} scale={[.82,.34,.42]} onClick={() => onSelect("Liver")}><sphereGeometry args={[.58, 36, 36]} /><meshStandardMaterial color={organ === "Liver" ? "#f2b84b" : "#bb8640"} transparent opacity={organ === "Liver" ? .48 : .23} emissive="#f2b84b" emissiveIntensity={.32} /></mesh>
    <mesh position={[0, -.5, -.02]} onClick={() => onSelect("Bone")}><cylinderGeometry args={[.035,.055,2.6,16]} /><meshStandardMaterial color={organ === "Bone" ? "#f2b84b" : "#9bcbd2"} transparent opacity={.55} emissive={colors.line} emissiveIntensity={.32} /></mesh>
    <mesh position={[0, .74, .1]}><cylinderGeometry args={[.055,.07,.92,18]} /><meshStandardMaterial color="#8ce8f5" transparent opacity={.58} /></mesh>
    <Line points={[[0,.35,.1],[-.35,.05,.1],[-.48,-.15,.12]]} color={colors.line} lineWidth={1.2} transparent opacity={.7} /><Line points={[[0,.35,.1],[.35,.05,.1],[.48,-.15,.12]]} color={colors.line} lineWidth={1.2} transparent opacity={.7} />
    {[-.55,-.28,.3].map((y,index) => <Line key={y} points={[[-.55,y,-.02],[0,y+.1,.14],[.55,y,-.02]]} color={index === 1 ? "#ff596a" : colors.line} lineWidth={.65} transparent opacity={.43} />)}
    <PulseTumor onSelect={() => onSelect("Lungs")} />
    <group onClick={() => onSelect("Nodes")}>{[[-.58,.43,.25],[-.48,.22,.3],[-.62,.05,.2]].map((position,index) => <mesh key={index} position={position as [number,number,number]}><sphereGeometry args={[.065,24,24]} /><meshStandardMaterial color="#916bff" emissive="#7655ff" emissiveIntensity={organ === "Nodes" ? 2 : .85} /></mesh>)}<Html position={[-.75,.48,.12]} center transform distanceFactor={4.4} className="three-label node-label"><b>LYMPH NODE</b><span>Involvement · 3 nodes</span></Html></group>
    <Html position={[.77,-.7,.08]} center transform distanceFactor={4.4} className="three-label risk-label"><b>METASTATIC RISK</b><span>Low</span></Html>
    <HolographicPlatform color={colors.line} /><SpatialParticles color={colors.line} />
  </group>;
}

function LoadingTwin() { return <div className="twin-loading"><span /><b>Initializing spatial patient model</b></div>; }

export function PatientDigitalTwin({ patientId }: { patientId: string }) {
  const [organ, setOrgan] = useState<Organ>("Lungs"), [mode, setMode] = useState<ViewMode>("3D View");
  const controls = useRef<OrbitControlsImpl>(null);
  const reset = () => { setOrgan("Lungs"); controls.current?.reset(); };
  const fullScreen = () => { document.querySelector(".digital-twin-card")?.requestFullscreen?.(); };
  const organIcons: Record<Organ, typeof ScanLine> = { Lungs: ScanLine, Brain, Liver: Activity, Bone, Nodes: Focus };
  return <section className="card digital-twin-card">
    <div className="twin-toolbar"><div><p className="eyebrow">3D DIGITAL TWIN</p><h2>Spatial disease intelligence</h2></div><div className="twin-actions"><span className="live-signal"><i /> WEBGL · {patientId}</span><button onClick={reset} aria-label="Reset digital twin view"><RotateCcw /></button><button onClick={fullScreen} aria-label="Open digital twin fullscreen"><Expand /></button></div></div>
    <div className="twin-modes" role="tablist" aria-label="Visualization modes">{(["3D View","CT Scan","Pathology","Gene Map"] as ViewMode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)} role="tab" aria-selected={mode === item}>{item}</button>)}</div>
    <div className="twin-stage"><Suspense fallback={<LoadingTwin />}><Canvas camera={{ position: [0,.55,4.3], fov: 38, near: .1, far: 100 }} dpr={[1,1.5]} gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}>
      <color attach="background" args={["#020a11"]} /><fog attach="fog" args={["#020a11", 5.5, 10]} /><ambientLight intensity={.7} /><pointLight color="#35d6f2" position={[2,3,4]} intensity={18} distance={9} /><pointLight color="#f2b84b" position={[-3,1,2]} intensity={10} distance={8} />
      <AnatomyModel organ={organ} mode={mode} onSelect={setOrgan} /><OrbitControls ref={controls} makeDefault enablePan={false} enableDamping dampingFactor={.075} minDistance={3.3} maxDistance={7} minPolarAngle={Math.PI * .26} maxPolarAngle={Math.PI * .72} /><CameraController organ={organ} controls={controls} /><AdaptiveDpr pixelated />
    </Canvas></Suspense>
      <div className="organ-selector vertical" aria-label="Choose organ focus">{(Object.keys(organIcons) as Organ[]).map((item) => { const Icon = organIcons[item] as typeof ScanLine; return <button key={item} className={organ === item ? "active" : ""} onClick={() => setOrgan(item)}><Icon /><span>{item}</span></button>; })}</div>
      <div className="twin-mode-readout"><span>{mode}</span><b>{organ} focus</b></div><small className="twin-hint">Drag to rotate · Scroll to zoom · Double-click anatomy to focus</small>
    </div>
    <div className="twin-footer-controls">{(["3D View","CT Scan","Pathology","Gene Map"] as ViewMode[]).map((item) => <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)}>{item}</button>)}<button onClick={reset}><RotateCcw /> Reset View</button></div>
  </section>;
}
