import React, { useEffect, useState, useMemo } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Play, Pause } from 'lucide-react';
import { AnimationPlayer3D } from './AnimationPlayer3D';
import { resolveAnimationForPhrase } from '../api/animationApi';

export const AvatarPlayer = ({ text, language, isAnimating, onAnimationStart, onAnimationEnd }) => {
  const { t } = useLanguage();
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAnimation, setCurrentAnimation] = useState(null);
  const [loadingAnimation, setLoadingAnimation] = useState(false);

  // Memoize words array to prevent infinite re-renders
  const words = useMemo(() => {
    return text ? text.split(' ').filter(w => w.length > 0) : [];
  }, [text]);

  // Load default animation on mount (show model immediately)
  useEffect(() => {
    const loadDefaultAnimation = async () => {
      try {
        const result = await resolveAnimationForPhrase('ready', 'psl');
        setCurrentAnimation(result.animation);
      } catch (error) {
        console.error('Failed to load default animation:', error);
      }
    };

    if (!text && !currentAnimation) {
      loadDefaultAnimation();
    }
  }, []);

  // Reset state when text changes (handles interruption)
  useEffect(() => {
    setCurrentWordIndex(0);
    if (text) {
      setCurrentAnimation(null);
    }
  }, [text]);

  // Load animation for current word (only when word index or text changes)
  useEffect(() => {
    if (words.length > 0 && currentWordIndex < words.length && !loadingAnimation) {
      const currentWord = words[currentWordIndex];
      loadAnimationForWord(currentWord);
    }
  }, [currentWordIndex, text]); // Depend on text, not words array

  // Auto-play when isAnimating prop changes
  useEffect(() => {
    if (isAnimating && words.length > 0) {
      handlePlay();
    } else if (!isAnimating) {
      setIsPlaying(false);
    }
  }, [isAnimating]);

  const loadAnimationForWord = async (word) => {
    setLoadingAnimation(true);
    try {
      const result = await resolveAnimationForPhrase(word, language);
      setCurrentAnimation(result.animation);
      console.log(`Loaded animation for "${word}":`, result.animation);
    } catch (error) {
      console.error(`Failed to load animation for "${word}":`, error);
      setCurrentAnimation(null);
    } finally {
      setLoadingAnimation(false);
    }
  };

  const handlePlay = () => {
    setCurrentWordIndex(0);
    setIsPlaying(true);
    onAnimationStart?.();
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleAnimationComplete = () => {
    console.log(`Animation complete for word ${currentWordIndex + 1}/${words.length}`);

    // Move to next word after animation completes
    setTimeout(() => {
      if (currentWordIndex < words.length - 1) {
        setCurrentWordIndex(prev => prev + 1);
      } else {
        // All words complete
        setIsPlaying(false);
        onAnimationEnd?.();
        setCurrentWordIndex(0);
      }
    }, 500); // Small delay between words
  };

  const currentWord = words[currentWordIndex] || '';

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* 3D Avatar Display Area */}
      <div className="relative rounded-lg aspect-video overflow-hidden">
        {currentAnimation ? (
          <>
            <AnimationPlayer3D
              key={`${currentAnimation.id}-${currentWordIndex}`}
              animationPath={currentAnimation.file_path}
              isPlaying={isPlaying}
              animationSpeed={1.0}
              onAnimationComplete={handleAnimationComplete}
            />
            {/* Current Word Overlay - only show when text is being animated */}
            {text && (
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
                <div className="bg-voice-green/90 text-white px-6 py-2 rounded-full font-semibold text-lg shadow-lg">
                  {currentWord}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="bg-gradient-to-b from-blue-50 to-blue-100 rounded-lg h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-sm mt-2">
                {loadingAnimation ? 'Loading 3D model...' : 'Preparing avatar...'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Animation Progress */}
      {words.length > 0 && (
        <div className="text-sm text-gray-600 text-center">
          <p>
            {currentWordIndex + 1} / {words.length} words
          </p>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className="bg-voice-green h-2 rounded-full transition-all duration-300"
              style={{
                width: `${((currentWordIndex + 1) / words.length) * 100}%`
              }}
            />
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-3">
        {!isPlaying ? (
          <button
            onClick={handlePlay}
            disabled={!text || loadingAnimation}
            className="flex-1 flex items-center justify-center gap-2 bg-voice-green text-white px-4 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-5 h-5" />
            {t('text_replay')}
          </button>
        ) : (
          <button
            onClick={handlePause}
            className="flex-1 flex items-center justify-center gap-2 bg-orange-500 text-white px-4 py-3 rounded-lg font-medium hover:bg-orange-600 transition-colors"
          >
            <Pause className="w-5 h-5" />
            Pause
          </button>
        )}
      </div>

      {/* Word List */}
      {words.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-700 uppercase">
            Words to animate
          </h4>
          <div className="flex flex-wrap gap-2">
            {words.map((word, idx) => (
              <span
                key={idx}
                className={`px-3 py-1 rounded-full text-sm font-medium transition-all ${
                  idx === currentWordIndex
                    ? 'bg-voice-green text-white scale-110'
                    : idx < currentWordIndex
                    ? 'bg-green-200 text-green-800'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {word}
              </span>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};
