"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function Globe() {
  const globe = useRef<THREE.Group>(null);
  const locations = useMemo(() => [[.55,.45,.78],[-.68,.58,.5],[.78,-.12,.6],[-.5,-.42,.72],[.18,-.68,.72],[-.8,.02,.4]] as [number,number,number][], []);
  useFrame((_, delta) => { if (!document.hidden && globe.current) globe.current.rotation.y += delta * .08; });
  return <group ref={globe} rotation={[.15,-.4,0]}>
    <mesh><sphereGeometry args={[1,32,32]} /><meshPhysicalMaterial color="#0b6b83" transparent opacity={.24} roughness={.4} metalness={.35} emissive="#0d8da8" emissiveIntensity={.25} /></mesh>
    <mesh><sphereGeometry args={[1.012,20,20]} /><meshBasicMaterial color="#35d6f2" wireframe transparent opacity={.16} /></mesh>
    {locations.map((position,index) => <group key={index} position={position}><mesh><sphereGeometry args={[.035,12,12]} /><meshBasicMaterial color={index === 2 ? "#f2b84b" : "#35d6f2"} /></mesh><pointLight color="#35d6f2" intensity={1.4} distance={.45} /></group>)}
    <mesh rotation={[Math.PI/2,0,0]}><torusGeometry args={[1.2,.008,8,96]} /><meshBasicMaterial color="#35d6f2" transparent opacity={.35} /></mesh>
  </group>;
}

export function GlobalImpactGlobe() {
  return <div className="global-impact-card card">
    <div className="impact-globe" data-webgl="global-impact"><Canvas camera={{ position: [0,0,3.1], fov: 40 }} dpr={[1,1.25]} gl={{ alpha:true,antialias:true,powerPreference:"high-performance" }}><ambientLight intensity={.65}/><pointLight position={[3,2,4]} color="#35d6f2" intensity={12}/><Globe/><OrbitControls enablePan={false} enableZoom={false} rotateSpeed={.35}/></Canvas></div>
    <div className="impact-copy"><p className="eyebrow">GLOBAL IMPACT</p><div><span><b>12+</b>Countries</span><span><b>2,500+</b>Patients analyzed</span><span><b>94%</b>Model accuracy</span></div><small>Demo network · synthetic aggregate metrics</small></div>
  </div>;
}
