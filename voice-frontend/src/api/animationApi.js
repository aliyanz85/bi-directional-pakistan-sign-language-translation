import axios from 'axios';

// Get base URL from environment variable or use default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

/**
 * Health check - verify backend is running
 * @returns {Promise<Object>} Health status object
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Get all available animations
 * @returns {Promise<Array>} List of animation objects with metadata
 */
export const getAnimations = async () => {
  try {
    const response = await apiClient.get('/api/animations');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch animations:', error);
    throw error;
  }
};

/**
 * Get a specific animation by ID
 * @param {string} animationId - The animation ID (e.g., 'hello', 'thanks')
 * @returns {Promise<Object>} Animation object with metadata
 */
export const getAnimationById = async (animationId) => {
  try {
    const response = await apiClient.get(`/api/animations/${animationId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch animation ${animationId}:`, error);
    throw error;
  }
};

/**
 * Resolve the best animation for a given phrase
 * @param {string} phrase - The text phrase to translate
 * @param {string} language - The language code (default: 'psl')
 * @returns {Promise<Object>} Object with animation, matched_word, and confidence
 */
export const resolveAnimationForPhrase = async (phrase, language = 'psl') => {
  try {
    const response = await apiClient.post('/api/resolve-animation', {
      phrase,
      language,
    });
    return response.data;
  } catch (error) {
    console.error(`Failed to resolve animation for phrase "${phrase}":`, error);
    throw error;
  }
};

/**
 * Get animations by category
 * @param {string} category - The category to filter by
 * @returns {Promise<Array>} Filtered list of animations
 */
export const getAnimationsByCategory = async (category) => {
  try {
    const animations = await getAnimations();
    return animations.filter(anim => anim.category === category);
  } catch (error) {
    console.error(`Failed to fetch animations by category ${category}:`, error);
    throw error;
  }
};

/**
 * Search animations by tag
 * @param {string} tag - The tag to search for
 * @returns {Promise<Array>} Animations matching the tag
 */
export const searchAnimationsByTag = async (tag) => {
  try {
    const animations = await getAnimations();
    return animations.filter(anim =>
      anim.tags.some(t => t.toLowerCase().includes(tag.toLowerCase()))
    );
  } catch (error) {
    console.error(`Failed to search animations by tag ${tag}:`, error);
    throw error;
  }
};

/**
 * Get Urdu word mappings for animations
 * @returns {Promise<Object>} Dictionary of English to Urdu mappings
 */
export const getUrduWords = async () => {
  try {
    const response = await apiClient.get('/api/urdu-words');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch Urdu words:', error);
    throw error;
  }
};

export default {
  checkHealth,
  getAnimations,
  getAnimationById,
  resolveAnimationForPhrase,
  getAnimationsByCategory,
  searchAnimationsByTag,
  getUrduWords,
};
