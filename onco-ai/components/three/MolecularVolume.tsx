"use client";
import { Line } from "@react-three/drei";

export function MolecularVolume({ pathology }: { pathology: boolean }) {
  if (pathology) return <group>{Array.from({length:65},(_,i) => {
    const angle=i*2.399, r=.13*Math.sqrt(i);
    return <group key={i} position={[Math.cos(angle)*r,Math.sin(angle)*r,.2*Math.sin(i)]}>
      <mesh><icosahedronGeometry args={[.14+(i%4)*.01,2]}/><meshStandardMaterial color={i%5===0 ? '#e88fb6' : '#8b6fd8'} transparent opacity={.5} roughness={.4}/></mesh>
      <mesh><icosahedronGeometry args={[.055,2]}/><meshBasicMaterial color="#cbb4ff"/></mesh>
    </group>;
  })}</group>;
  const strand = (offset:number) => Array.from({length:100},(_,i) => {
    const a=i*.16+offset;
    return [Math.cos(a)*.5,i*.026-1.3,Math.sin(a)*.5] as [number,number,number];
  });
  return <group rotation={[0,0,-.15]}>
    <Line points={strand(0)} color="#43dbff" lineWidth={5}/><Line points={strand(Math.PI)} color="#b179ff" lineWidth={5}/>
    {Array.from({length:25},(_,i) => {const a=i*.64;return <Line key={i} points={[[Math.cos(a)*.5,i*.104-1.3,Math.sin(a)*.5],[-Math.cos(a)*.5,i*.104-1.3,-Math.sin(a)*.5]]} color={i%3===0?'#f4ce75':'#60e5dd'} lineWidth={2}/>;})}
  </group>;
}
