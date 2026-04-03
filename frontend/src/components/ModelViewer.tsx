"use client";

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF, Center, Grid } from "@react-three/drei";
import * as THREE from "three";

interface ModelViewerProps {
  url: string;
  className?: string;
}

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const ref = useRef<THREE.Group>(null);

  // Slow auto-rotation
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <Center>
      <group ref={ref}>
        <primitive object={scene} />
      </group>
    </Center>
  );
}

function LoadingFallback() {
  const ref = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.x += delta * 0.5;
      ref.current.rotation.y += delta * 0.7;
    }
  });

  return (
    <mesh ref={ref}>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial color="#4c6ef5" wireframe />
    </mesh>
  );
}

export default function ModelViewer({ url, className = "" }: ModelViewerProps) {
  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden ${className}`}>
      <Canvas
        camera={{ position: [2, 1.5, 2], fov: 50 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        style={{ background: "#111" }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />

        <Suspense fallback={<LoadingFallback />}>
          <Model url={url} />
          <Environment preset="studio" />
        </Suspense>

        <OrbitControls
          makeDefault
          autoRotate={false}
          enablePan={true}
          enableZoom={true}
          minDistance={0.5}
          maxDistance={10}
        />

        <Grid
          position={[0, -0.5, 0]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#333"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#444"
          fadeDistance={10}
          infiniteGrid
        />
      </Canvas>

      {/* Controls hint */}
      <div className="absolute bottom-2 right-2 text-[10px] text-gray-500 bg-black/50 px-2 py-1 rounded">
        Drag: rotar | Scroll: zoom | Shift+drag: mover
      </div>
    </div>
  );
}
