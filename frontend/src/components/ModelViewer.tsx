"use client";

import { Suspense, useRef, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, useGLTF, useAnimations, Center, Grid } from "@react-three/drei";
import * as THREE from "three";

interface ModelViewerProps {
  url: string;
  className?: string;
  autoRotate?: boolean;
}

function AnimatedModel({ url }: { url: string }) {
  const group = useRef<THREE.Group>(null);
  const { scene, animations } = useGLTF(url);
  const { actions } = useAnimations(animations, group);

  // Play all animations on load
  useEffect(() => {
    if (actions && Object.keys(actions).length > 0) {
      Object.values(actions).forEach((action) => {
        if (action) {
          action.reset().fadeIn(0.5).play();
        }
      });
    }
  }, [actions]);

  // Slow auto-rotation only when no animations
  const hasAnimations = animations.length > 0;

  useFrame((_, delta) => {
    if (group.current && !hasAnimations) {
      group.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <Center>
      <group ref={group}>
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

export default function ModelViewer({ url, className = "", autoRotate = false }: ModelViewerProps) {
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
          <AnimatedModel url={url} />
          <Environment preset="studio" />
        </Suspense>

        <OrbitControls
          makeDefault
          autoRotate={autoRotate}
          enablePan={true}
          enableZoom={true}
          minDistance={0.5}
          maxDistance={20}
        />

        <Grid
          position={[0, -0.5, 0]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#333"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#444"
          fadeDistance={15}
          infiniteGrid
        />
      </Canvas>

      <div className="absolute bottom-2 right-2 text-[10px] text-gray-500 bg-black/50 px-2 py-1 rounded">
        Drag: rotar | Scroll: zoom | Shift+drag: mover
      </div>
    </div>
  );
}
