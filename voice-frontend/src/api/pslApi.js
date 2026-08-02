/**
 * PSL Recognition API Service
 *
 * Handles communication with the backend PSL recognition endpoints.
 */

import axios from 'axios';

// Get API base URL from environment variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with base configuration
const pslApiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for model inference
});

/**
 * Recognize PSL sign from a sequence of landmark features
 *
 * @param {number[][]} sequence - Array of 60 frames, each with 188 features
 * @returns {Promise<{label: string, class_id: number, confidence: number, top_predictions: Array}>}
 * @throws {Error} If the request fails or response is invalid
 */
export const recognizePSL = async (sequence) => {
  try {
    console.log('Sending PSL recognition request...', {
      sequenceLength: sequence.length,
      featureLength: sequence[0]?.length
    });

    const response = await pslApiClient.post('/api/psl/recognize', {
      sequence
    });

    console.log('PSL recognition response:', response.data);
    return response.data;

  } catch (error) {
    console.error('PSL recognition error:', error);

    if (error.response) {
      // Server responded with error status
      const errorMessage = error.response.data?.detail || 'Recognition failed';
      throw new Error(`Server error: ${errorMessage}`);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Is the backend running?');
    } else {
      // Something else happened
      throw new Error(`Request error: ${error.message}`);
    }
  }
};

/**
 * Get PSL model information
 *
 * @returns {Promise<Object>} Model metadata
 */
export const getPSLModelInfo = async () => {
  try {
    const response = await pslApiClient.get('/api/psl/model-info');
    return response.data;
  } catch (error) {
    console.error('Error fetching PSL model info:', error);
    throw error;
  }
};

/**
 * Check PSL service health
 *
 * @returns {Promise<Object>} Health status
 */
export const checkPSLHealth = async () => {
  try {
    const response = await pslApiClient.get('/api/psl/health');
    return response.data;
  } catch (error) {
    console.error('Error checking PSL health:', error);
    throw error;
  }
};

/**
 * Validate a sequence before sending to the backend
 *
 * @param {number[][]} sequence - Sequence to validate
 * @returns {{valid: boolean, error: string|null}}
 */
export const validateSequence = (sequence) => {
  if (!Array.isArray(sequence)) {
    return { valid: false, error: 'Sequence must be an array' };
  }

  if (sequence.length !== 60) {
    return { valid: false, error: `Sequence must have 60 frames, got ${sequence.length}` };
  }

  for (let i = 0; i < sequence.length; i++) {
    const frame = sequence[i];

    if (!Array.isArray(frame)) {
      return { valid: false, error: `Frame ${i} must be an array` };
    }

    if (frame.length !== 188) {
      return { valid: false, error: `Frame ${i} must have 188 features, got ${frame.length}` };
    }

    // Check for NaN or Infinity
    for (let j = 0; j < frame.length; j++) {
      if (typeof frame[j] !== 'number' || !isFinite(frame[j])) {
        return { valid: false, error: `Frame ${i}, feature ${j} is not a valid number` };
      }
    }
  }

  return { valid: true, error: null };
};

export default {
  recognizePSL,
  getPSLModelInfo,
  checkPSLHealth,
  validateSequence
};
