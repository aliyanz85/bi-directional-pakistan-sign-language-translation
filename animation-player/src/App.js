import React, { useState, Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import * as THREE from 'three';
import './App.css';

// List of all 32 words with their animations
const WORDS = [
  'alert', 'book', 'careful', 'cheap', 'crazy', 'dangerous', 'decent', 'dumb',
  'excited', 'extreme', 'fantastic', 'far', 'fearful', 'foreign', 'funny', 'good',
  'healthy', 'heavy', 'important', 'intelligent', 'interesting', 'late', 'less', 'new',
  'no', 'noisy', 'peaceful', 'quick', 'ready', 'secure', 'smart', 'yes'
];

// Component to load and display the GLB model with animations
function Model({ word, isPlaying, animationSpeed }) {
  const group = useRef();
  const mixerRef = useRef();
  const actionsRef = useRef([]);
  
  const gltf = useLoader(GLTFLoader, `/animations/${word}.glb`);

  useEffect(() => {
    // Reset previous animations
    if (mixerRef.current) {
      mixerRef.current.stopAllAction();
    }
    
    if (gltf.animations && gltf.animations.length > 0) {
      mixerRef.current = new THREE.AnimationMixer(gltf.scene);
      actionsRef.current = gltf.animations.map(clip => 
        mixerRef.current.clipAction(clip)
      );
    }
  }, [gltf, word]);

  useEffect(() => {
    if (actionsRef.current.length > 0) {
      actionsRef.current.forEach(action => {
        action.timeScale = animationSpeed;
        if (isPlaying) {
          action.reset().fadeIn(0.5).play();
        } else {
          action.fadeOut(0.5);
        }
      });
    }
  }, [isPlaying, animationSpeed]);

  useFrame((state, delta) => {
    if (mixerRef.current && isPlaying) {
      mixerRef.current.update(delta);
    }
  });

  return (
    <group ref={group}>
      <primitive object={gltf.scene} scale={1} position={[0, -1, 0]} />
    </group>
  );
}

// Loading spinner component
function Loader() {
  return (
    <div className="loader">
      <div className="spinner"></div>
      <p>Loading 3D Model...</p>
    </div>
  );
}

// OrbitControls component
function CameraControls() {
  const { camera, gl } = useThree();
  const controlsRef = useRef();
  
  useEffect(() => {
    controlsRef.current = new OrbitControls(camera, gl.domElement);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.05;
    
    return () => {
      controlsRef.current.dispose();
    };
  }, [camera, gl]);

  useFrame(() => {
    if (controlsRef.current) {
      controlsRef.current.update();
    }
  });

  return null;
}

function App() {
  const [selectedWord, setSelectedWord] = useState('alert');
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [key, setKey] = useState(0); // Force re-render when word changes

  const handleWordSelect = (word) => {
    setIsPlaying(false);
    setSelectedWord(word);
    setKey(prev => prev + 1); // Force model reload
  };

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSpeedChange = (e) => {
    setAnimationSpeed(parseFloat(e.target.value));
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎬 Sign Language Animation Player</h1>
        <p>Select a word to see its sign language animation</p>
      </header>

      <div className="main-content">
        <div className="word-selection">
          <h2>Select a Word ({WORDS.length} words)</h2>
          <div className="word-grid">
            {WORDS.map(word => (
              <button
                key={word}
                className={`word-button ${selectedWord === word ? 'selected' : ''}`}
                onClick={() => handleWordSelect(word)}
              >
                {word}
              </button>
            ))}
          </div>
        </div>

        <div className="viewer-section">
          <div className="selected-word-display">
            <span>Currently showing:</span>
            <strong>{selectedWord}</strong>
          </div>

          <div className="canvas-container">
            <Suspense fallback={<Loader />}>
              <Canvas
                key={key}
                camera={{ position: [0, 2, 5], fov: 50 }}
                shadows
              >
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
                <pointLight position={[-10, -10, -5]} intensity={0.5} />
                <Model 
                  word={selectedWord} 
                  isPlaying={isPlaying} 
                  animationSpeed={animationSpeed} 
                />
                <CameraControls />
              </Canvas>
            </Suspense>
          </div>

          <div className="controls">
            <button 
              className={`play-button ${isPlaying ? 'playing' : ''}`}
              onClick={handlePlayPause}
            >
              {isPlaying ? '⏸ Pause Animation' : '▶ Play Animation'}
            </button>

            <div className="speed-control">
              <label htmlFor="speed">Animation Speed: {animationSpeed}x</label>
              <input
                type="range"
                id="speed"
                min="0.1"
                max="3"
                step="0.1"
                value={animationSpeed}
                onChange={handleSpeedChange}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
