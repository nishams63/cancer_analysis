"use client";

import { useEffect, useMemo, useState } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";

// Procedural educational volume, not a segmentation of the patient's imaging.
function lungGeometry(side: number) {
  const geometry = new THREE.SphereGeometry(1, 48, 48);
  const positions = geometry.attributes.position;
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
    const width = .37 * (.85 - .22 * y);
    const hilum = Math.max(0, -side * x) * Math.exp(-y * y * 7) * .11;
    positions.setXYZ(i, x * width + side * hilum, y * .79, z * .28 * (1 - .14 * y));
  }
  geometry.computeVertexNormals();
  return geometry;
}

function branches(side: number) {
  const paths: THREE.TubeGeometry[] = [];
  function grow(start: THREE.Vector3, end: THREE.Vector3, depth: number, seed: number) {
    const middle = start.clone().lerp(end, .5).add(new THREE.Vector3(side * .025, .018, .025));
    paths.push(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([start, middle, end]), 8, .012 * (depth + 1) / 4, 5, false));
    if (!depth) return;
    for (let j = 0; j < 3; j++) {
      const a = seed * 1.91 + j * 2.1;
      const length = .13 + depth * .028;
      const next = end.clone().add(new THREE.Vector3(side * (.025 + Math.abs(Math.cos(a)) * length), (j - 1) * length, Math.sin(a) * length * .65));
      next.x = side * Math.min(.77, Math.abs(next.x));
      grow(end, next, depth - 1, seed * 3 + j + 1);
    }
  }
  for (let i = 0; i < 5; i++) grow(new THREE.Vector3(side * .06, .95, 0), new THREE.Vector3(side * .3, .85 - i * .23, .02), 3, i + 1);
  return paths;
}

export function ThoracicVolume({ color, onSelect, ct = false }: { color: string; onSelect: () => void; ct?: boolean }) {
  const [hovered, setHovered] = useState(false);
  const geometries = useMemo(() => [-1, 1].map(side => ({ side, lung: lungGeometry(side), airways: branches(side) })), []);
  useEffect(() => () => geometries.forEach(g => { g.lung.dispose(); g.airways.forEach(a => a.dispose()); }), [geometries]);
  return <group onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)} onClick={e => { e.stopPropagation(); onSelect(); }} onDoubleClick={e => { e.stopPropagation(); onSelect(); }}>
    {geometries.map(({ side, lung, airways }) => <group key={side}>
      <mesh geometry={lung} position={[side * .4, .4, 0]}>
        <meshPhysicalMaterial color={color} transparent opacity={hovered ? .33 : .21} roughness={.3} emissive={color} emissiveIntensity={.45} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh geometry={lung} position={[side * .4, .4, 0]}><meshBasicMaterial color={color} wireframe transparent opacity={.065} depthWrite={false} /></mesh>
      {airways.map((geometry, i) => <mesh key={i} geometry={geometry}><meshBasicMaterial color={i % 4 === 0 ? "#b4f1ff" : color} transparent opacity={.4} /></mesh>)}
      {Array.from({ length: 10 }, (_, i) => {
        const y = 1.12 - i * .145, width = .54 + Math.sin(i / 10 * Math.PI) * .29;
        const points = Array.from({ length: 30 }, (_, j) => {
          const angle = j / 29 * Math.PI;
          return [side * Math.sin(angle) * width, y - Math.sin(angle) * .14, -.17 + (1 - Math.cos(angle)) * .23] as [number, number, number];
        });
        return <Line key={i} points={points} color={ct ? "#e1edf8" : "#48b9ec"} lineWidth={1.6} transparent opacity={.32} />;
      })}
      <Line points={[[side * .08,1.43,0],[side*.35,1.4,.04],[side*.7,1.27,0],[side*.89,1.05,-.04],[side*.97,.6,-.1],[side*.84,-.42,-.1],[side*.66,-.92,-.1]]} color={color} lineWidth={1} transparent opacity={.45} />
    </group>)}
    {Array.from({ length: 21 }, (_, i) => <mesh key={i} position={[0,1.43-i*.115,-.19]} rotation={[.05,0,0]}><boxGeometry args={[.105,.07,.105]} /><meshStandardMaterial color="#7cc7ec" emissive="#217eab" emissiveIntensity={.5} transparent opacity={.55}/></mesh>)}
    <Line points={[[0,1.57,.03],[0,1.07,.03],[-.18,.86,.02],[-.3,.65,.01]]} color="#a4e4f6" lineWidth={5} transparent opacity={.6}/>
    <Line points={[[0,1.07,.03],[.18,.86,.02],[.3,.65,.01]]} color="#a4e4f6" lineWidth={4} transparent opacity={.6}/>
    {ct && [-.4,0,.4,.8].map(y => <mesh key={y} position={[0,y,0]} rotation={[-Math.PI/2,0,0]}><planeGeometry args={[1.8,.9]}/><meshBasicMaterial color="#b7edff" transparent opacity={.09} side={THREE.DoubleSide}/></mesh>)}
  </group>;
}
