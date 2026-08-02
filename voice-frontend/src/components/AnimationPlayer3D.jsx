import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import * as THREE from 'three';

/**
 * 3D Model component that loads and animates GLB files
 */
function Model({ animationPath, isPlaying, animationSpeed, onAnimationComplete }) {
  const group = useRef();
  const mixerRef = useRef();
  const actionsRef = useRef([]);
  const hasCompletedRef = useRef(false);

  // Load the GLB model
  const gltf = useLoader(GLTFLoader, animationPath);

  // Setup animation mixer when model loads
  useEffect(() => {
    console.log('Model loaded:', animationPath);
    console.log('Animations found:', gltf.animations.length);

    // Reset previous animations
    if (mixerRef.current) {
      mixerRef.current.stopAllAction();
    }

    hasCompletedRef.current = false;

    if (gltf.animations && gltf.animations.length > 0) {
      mixerRef.current = new THREE.AnimationMixer(gltf.scene);

      // Create actions for all animation clips
      actionsRef.current = gltf.animations.map(clip => {
        console.log('Creating action for clip:', clip.name, 'duration:', clip.duration);
        const action = mixerRef.current.clipAction(clip);
        action.setLoop(THREE.LoopOnce); // Play once
        action.clampWhenFinished = true; // Stay on last frame
        return action;
      });

      // Listen for animation finish
      mixerRef.current.addEventListener('finished', () => {
        console.log('Animation finished event fired');
        if (!hasCompletedRef.current) {
          hasCompletedRef.current = true;
          onAnimationComplete?.();
        }
      });

      // If isPlaying is already true, start playing immediately
      if (isPlaying) {
        console.log('Auto-starting animation (isPlaying was true on load)');
        actionsRef.current.forEach(action => {
          action.timeScale = animationSpeed;
          action.reset().play();
        });
      }
    } else {
      console.warn('No animations found in GLB file:', animationPath);
    }

    return () => {
      if (mixerRef.current) {
        mixerRef.current.stopAllAction();
      }
    };
  }, [gltf, animationPath]);

  // Control animation playback when isPlaying changes
  useEffect(() => {
    if (actionsRef.current.length > 0) {
      actionsRef.current.forEach(action => {
        action.timeScale = animationSpeed;
        if (isPlaying) {
          console.log('Starting animation playback');
          action.reset().play();
        } else {
          console.log('Stopping animation playback');
          action.stop();
        }
      });
    }
  }, [isPlaying, animationSpeed]);

  // Update animation mixer on each frame
  useFrame((state, delta) => {
    if (mixerRef.current && isPlaying) {
      mixerRef.current.update(delta);
    }
  });

  // Apply better materials to the model
  useEffect(() => {
    if (gltf.scene) {
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          // Ensure the mesh casts and receives shadows
          child.castShadow = true;
          child.receiveShadow = true;

          // Enhance material appearance
          if (child.material) {
            child.material.metalness = 0.1;
            child.material.roughness = 0.6;
            child.material.envMapIntensity = 1.5;

            // Enable better rendering
            if (child.material.map) {
              child.material.map.anisotropy = 16;
            }
          }
        }
      });
    }
  }, [gltf]);

  return (
    <group ref={group}>
      <primitive object={gltf.scene} scale={1.5} position={[0, -1.2, 0]} />
    </group>
  );
}

/**
 * Loading spinner component
 */
function Loader() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-voice-light border-t-voice-green rounded-full animate-spin mx-auto mb-3"></div>
        <p className="text-gray-600 text-sm">Loading 3D Model...</p>
      </div>
    </div>
  );
}

/**
 * Camera controls component for orbit interaction
 */
function CameraControls() {
  const { camera, gl } = useThree();
  const controlsRef = useRef();

  useEffect(() => {
    controlsRef.current = new OrbitControls(camera, gl.domElement);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.05;
    controlsRef.current.minDistance = 2;
    controlsRef.current.maxDistance = 6;
    controlsRef.current.target.set(0, 0.5, 0); // Focus on upper body
    controlsRef.current.maxPolarAngle = Math.PI / 1.5; // Limit vertical rotation

    return () => {
      controlsRef.current?.dispose();
    };
  }, [camera, gl]);

  useFrame(() => {
    controlsRef.current?.update();
  });

  return null;
}

/**
 * Main 3D Animation Player Component
 *
 * Renders a 3D avatar animation from a GLB file using Three.js and React Three Fiber
 *
 * @param {string} animationPath - Path to the GLB animation file (e.g., '/animations/hello.glb')
 * @param {boolean} isPlaying - Whether the animation should be playing
 * @param {number} animationSpeed - Playback speed multiplier (default: 1.0)
 * @param {function} onAnimationComplete - Callback when animation finishes
 * @param {string} className - Additional CSS classes
 */
export const AnimationPlayer3D = ({
  animationPath,
  isPlaying = false,
  animationSpeed = 1.0,
  onAnimationComplete,
  className = '',
}) => {
  const [error, setError] = useState(null);
  const [key, setKey] = useState(0);

  // Force reload when animation path changes
  useEffect(() => {
    console.log('AnimationPlayer3D received new path:', animationPath, 'isPlaying:', isPlaying);
    setError(null);
    setKey(prev => prev + 1);
  }, [animationPath]);

  // Error boundary for model loading
  const handleError = (error) => {
    console.error('Error loading 3D model:', error);
    setError('Failed to load animation. Please try another word.');
  };

  if (!animationPath) {
    return (
      <div className={`relative w-full h-full bg-gradient-to-b from-blue-50 to-blue-100 rounded-lg flex items-center justify-center ${className}`}>
        <p className="text-gray-400 text-center">No animation selected</p>
      </div>
    );
  }

  return (
    <div className={`relative w-full h-full bg-gradient-to-b from-blue-50 to-blue-100 rounded-lg overflow-hidden ${className}`}>
      {error ? (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-red-600 px-4">
            <p className="font-semibold">⚠️ {error}</p>
          </div>
        </div>
      ) : (
        <Suspense fallback={<Loader />}>
          <Canvas
            key={key}
            camera={{ position: [0, 1.5, 3], fov: 45 }}
            shadows
            onError={handleError}
            gl={{ antialias: true, alpha: true }}
          >
            {/* Enhanced Lighting */}
            <ambientLight intensity={0.6} />
            <directionalLight
              position={[5, 5, 5]}
              intensity={1.2}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
            />
            <pointLight position={[-5, 3, -5]} intensity={0.4} color="#ffffff" />
            <pointLight position={[5, -3, 5]} intensity={0.3} color="#bbdefb" />
            <hemisphereLight intensity={0.5} groundColor="#ffffff" />

            {/* 3D Model */}
            <Model
              animationPath={animationPath}
              isPlaying={isPlaying}
              animationSpeed={animationSpeed}
              onAnimationComplete={onAnimationComplete}
            />

            {/* Camera Controls */}
            <CameraControls />
          </Canvas>
        </Suspense>
      )}
    </div>
  );
};

export default AnimationPlayer3D;
