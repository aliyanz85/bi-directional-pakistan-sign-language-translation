/**
 * usePSLRecognition Hook
 *
 * Manages PSL recognition state, sequence buffering, and backend communication.
 * Accumulates 60 frames of features and sends to backend for prediction.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { recognizePSL, validateSequence } from '../api/pslApi';

/**
 * usePSLRecognition Hook
 *
 * @param {Object} options - Configuration options
 * @param {number} options.sequenceLength - Number of frames to buffer (default: 60)
 * @param {number} options.confidenceThreshold - Minimum confidence to accept prediction (default: 0.6)
 * @param {number} options.cooldownMs - Cooldown between API calls (default: 1000ms)
 * @param {boolean} options.autoRecognize - Automatically recognize when buffer is full (default: true)
 * @returns {Object} Recognition state and control functions
 */
export const usePSLRecognition = (options = {}) => {
  const {
    sequenceLength = 60,
    confidenceThreshold = 0.6,
    cooldownMs = 1000,
    autoRecognize = true
  } = options;

  // State
  const [sequenceBuffer, setSequenceBuffer] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [sentence, setSentence] = useState([]);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [error, setError] = useState(null);
  const [bufferProgress, setBufferProgress] = useState(0);

  // Refs for cooldown and last prediction tracking
  const lastRecognitionTime = useRef(0);
  const lastPredictedWord = useRef(null);
  const recognitionCooldown = useRef(false);

  /**
   * Add a feature frame to the sequence buffer
   *
   * @param {number[]} features - 188-dimensional feature vector
   */
  const addFrame = useCallback((features) => {
    if (!features || features.length !== 188) {
      console.warn('Invalid features length:', features?.length);
      return;
    }

    setSequenceBuffer(prevBuffer => {
      // Add new frame
      const newBuffer = [...prevBuffer, features];

      // Keep only the last sequenceLength frames (rolling window)
      if (newBuffer.length > sequenceLength) {
        newBuffer.shift(); // Remove oldest frame
      }

      // Update progress
      setBufferProgress((newBuffer.length / sequenceLength) * 100);

      return newBuffer;
    });
  }, [sequenceLength]);

  /**
   * Perform PSL recognition on the current buffer
   */
  const recognize = useCallback(async () => {
    if (sequenceBuffer.length < sequenceLength) {
      setError(`Buffer not full. Need ${sequenceLength} frames, have ${sequenceBuffer.length}`);
      return null;
    }

    // Check cooldown
    const now = Date.now();
    if (recognitionCooldown.current || (now - lastRecognitionTime.current) < cooldownMs) {
      console.log('Recognition on cooldown, skipping...');
      return null;
    }

    recognitionCooldown.current = true;
    setIsRecognizing(true);
    setError(null);

    try {
      console.log('Running PSL recognition...');

      // Validate sequence before sending
      const validation = validateSequence(sequenceBuffer);
      if (!validation.valid) {
        throw new Error(validation.error);
      }

      // Call backend API
      const result = await recognizePSL(sequenceBuffer);

      console.log('Recognition result:', result);

      // Update current prediction
      setCurrentPrediction(result);

      // Add to sentence if confidence is high enough and it's a new word
      if (result.confidence >= confidenceThreshold) {
        if (result.label !== lastPredictedWord.current) {
          setSentence(prevSentence => [...prevSentence, result.label]);
          lastPredictedWord.current = result.label;
          console.log(`Added "${result.label}" to sentence`);
        }
      } else {
        console.log(`Confidence too low (${result.confidence.toFixed(2)} < ${confidenceThreshold})`);
      }

      lastRecognitionTime.current = Date.now();

      return result;

    } catch (err) {
      console.error('Recognition error:', err);
      setError(err.message);
      return null;

    } finally {
      setIsRecognizing(false);
      // Release cooldown after delay
      setTimeout(() => {
        recognitionCooldown.current = false;
      }, cooldownMs);
    }
  }, [sequenceBuffer, sequenceLength, confidenceThreshold, cooldownMs]);

  /**
   * Auto-recognize when buffer is full (if enabled)
   */
  useEffect(() => {
    if (autoRecognize && sequenceBuffer.length === sequenceLength && !isRecognizing) {
      recognize();
    }
  }, [sequenceBuffer.length, sequenceLength, autoRecognize, isRecognizing, recognize]);

  /**
   * Clear the current sentence
   */
  const clearSentence = useCallback(() => {
    setSentence([]);
    lastPredictedWord.current = null;
    console.log('Sentence cleared');
  }, []);

  /**
   * Reset the sequence buffer
   */
  const resetBuffer = useCallback(() => {
    setSequenceBuffer([]);
    setBufferProgress(0);
    console.log('Buffer reset');
  }, []);

  /**
   * Remove the last word from the sentence
   */
  const undoLastWord = useCallback(() => {
    setSentence(prevSentence => {
      if (prevSentence.length === 0) return prevSentence;
      const newSentence = prevSentence.slice(0, -1);
      lastPredictedWord.current = newSentence[newSentence.length - 1] || null;
      return newSentence;
    });
  }, []);

  /**
   * Manually add a word to the sentence
   */
  const addWord = useCallback((word) => {
    setSentence(prevSentence => [...prevSentence, word]);
    lastPredictedWord.current = word;
  }, []);

  /**
   * Get sentence as a string
   */
  const getSentenceText = useCallback(() => {
    return sentence.join(' ');
  }, [sentence]);

  return {
    // State
    sequenceBuffer,
    currentPrediction,
    sentence,
    isRecognizing,
    error,
    bufferProgress,
    bufferLength: sequenceBuffer.length,
    bufferFull: sequenceBuffer.length === sequenceLength,

    // Functions
    addFrame,
    recognize,
    clearSentence,
    resetBuffer,
    undoLastWord,
    addWord,
    getSentenceText
  };
};

export default usePSLRecognition;
