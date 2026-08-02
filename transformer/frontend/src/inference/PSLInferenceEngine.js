/**
 * PSL Inference Engine - Browser-based Real-time Recognition
 * Uses TensorFlow.js and MediaPipe for hand tracking and sign recognition
 */

import * as tf from '@tensorflow/tfjs';

class PSLInferenceEngine {
  constructor() {
    this.model = null;
    this.isProcessing = false;
    this.sequenceBuffer = [];
    this.maxSequenceLength = 60;
    this.featureDim = 188;
    this.classNames = [];
    this.config = null;
    this.isReady = false;
    
    // Performance tracking
    this.inferenceStats = {
      totalInferences: 0,
      totalTime: 0,
      avgLatency: 0
    };
    
    // Prediction smoothing
    this.predictionHistory = [];
    this.smoothingWindow = 5;
  }
  
  /**
   * Initialize the inference engine
   */
  async initialize(modelPath = '/model/web_model', configPath = '/model/config.json') {
    console.log('Initializing PSL Inference Engine...');
    
    try {
      // Set backend to WebGL for GPU acceleration
      await tf.setBackend('webgl');
      await tf.ready();
      console.log('TensorFlow.js backend:', tf.getBackend());
      
      // Load configuration
      console.log('Loading configuration...');
      const configResponse = await fetch(configPath);
      this.config = await configResponse.json();
      
      this.classNames = this.config.classes;
      this.maxSequenceLength = this.config.sequenceLength;
      this.featureDim = this.config.featureDim;
      
      console.log(`Configuration loaded: ${this.classNames.length} classes`);
      
      // Load model
      console.log('Loading TensorFlow.js model...');
      this.model = await tf.loadGraphModel(`${modelPath}/model.json`);
      console.log('Model loaded successfully');
      
      // Warmup inference
      console.log('Warming up model...');
      await this.warmup();
      
      this.isReady = true;
      console.log('Inference engine ready!');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize inference engine:', error);
      throw error;
    }
  }
  
  /**
   * Warmup the model with dummy data
   */
  async warmup(numRuns = 3) {
    const dummyInput = tf.zeros([1, this.maxSequenceLength, this.featureDim]);
    
    for (let i = 0; i < numRuns; i++) {
      const prediction = await this.model.predict(dummyInput);
      prediction.dispose();
    }
    
    dummyInput.dispose();
    console.log('Model warmup complete');
  }
  
  /**
   * Process a single frame of hand landmarks
   */
  async processFrame(landmarks) {
    if (!this.isReady || !landmarks) {
      return null;
    }
    
    // Add to sequence buffer
    this.sequenceBuffer.push(landmarks);
    
    // Maintain buffer size
    if (this.sequenceBuffer.length > this.maxSequenceLength) {
      this.sequenceBuffer.shift();
    }
    
    // Only predict when we have enough frames
    if (this.sequenceBuffer.length < this.maxSequenceLength) {
      return {
        status: 'buffering',
        progress: this.sequenceBuffer.length / this.maxSequenceLength,
        message: `Collecting frames: ${this.sequenceBuffer.length}/${this.maxSequenceLength}`
      };
    }
    
    // Avoid concurrent predictions
    if (this.isProcessing) {
      return null;
    }
    
    this.isProcessing = true;
    
    try {
      // Prepare input tensor
      const inputTensor = tf.tensor3d(
        [this.sequenceBuffer],
        [1, this.maxSequenceLength, this.featureDim]
      );
      
      // Apply normalization if configured
      let normalizedInput = inputTensor;
      if (this.config.normalization?.enabled) {
        normalizedInput = this.normalize(inputTensor);
      }
      
      // Inference with timing
      const startTime = performance.now();
      const predictions = await this.model.predict(normalizedInput);
      const inferenceTime = performance.now() - startTime;
      
      // Update stats
      this.inferenceStats.totalInferences++;
      this.inferenceStats.totalTime += inferenceTime;
      this.inferenceStats.avgLatency = 
        this.inferenceStats.totalTime / this.inferenceStats.totalInferences;
      
      // Get top-k predictions
      const topK = await this.getTopK(predictions, this.config.inference.topK);
      
      // Apply temporal smoothing
      const smoothed = this.smoothPredictions(topK);
      
      // Cleanup tensors
      inputTensor.dispose();
      if (normalizedInput !== inputTensor) {
        normalizedInput.dispose();
      }
      predictions.dispose();
      
      return {
        status: 'success',
        predictions: smoothed,
        latency: inferenceTime,
        avgLatency: this.inferenceStats.avgLatency,
        confidence: smoothed[0].confidence
      };
      
    } catch (error) {
      console.error('Inference error:', error);
      return {
        status: 'error',
        message: error.message
      };
    } finally {
      this.isProcessing = false;
    }
  }
  
  /**
   * Normalize input tensor using saved parameters
   */
  normalize(tensor) {
    if (!this.config.normalization?.mean || !this.config.normalization?.std) {
      return tensor;
    }
    
    const mean = tf.tensor(this.config.normalization.mean);
    const std = tf.tensor(this.config.normalization.std);
    
    const normalized = tensor.sub(mean).div(std);
    
    mean.dispose();
    std.dispose();
    
    return normalized;
  }
  
  /**
   * Get top-k predictions
   */
  async getTopK(predictions, k = 3) {
    const values = await predictions.data();
    const indices = Array.from(values)
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .slice(0, k);
    
    return indices.map(item => ({
      class: this.classNames[item.idx],
      classIndex: item.idx,
      confidence: item.val
    }));
  }
  
  /**
   * Smooth predictions over time using exponential moving average
   */
  smoothPredictions(currentPredictions) {
    // Add to history
    this.predictionHistory.push(currentPredictions);
    
    // Maintain window size
    if (this.predictionHistory.length > this.smoothingWindow) {
      this.predictionHistory.shift();
    }
    
    // If not enough history, return current
    if (this.predictionHistory.length < 3) {
      return currentPredictions;
    }
    
    // Calculate weighted average (more recent = higher weight)
    const weights = this.predictionHistory.map((_, i) => i + 1);
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    
    // Aggregate predictions
    const aggregated = {};
    
    this.predictionHistory.forEach((preds, histIdx) => {
      preds.forEach(pred => {
        if (!aggregated[pred.class]) {
          aggregated[pred.class] = 0;
        }
        aggregated[pred.class] += pred.confidence * weights[histIdx] / totalWeight;
      });
    });
    
    // Sort and return top-k
    const sorted = Object.entries(aggregated)
      .map(([className, confidence]) => ({
        class: className,
        confidence: confidence,
        classIndex: this.classNames.indexOf(className)
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, this.config.inference.topK);
    
    return sorted;
  }
  
  /**
   * Check if prediction is stable
   */
  isPredictionStable(threshold = 0.7) {
    if (this.predictionHistory.length < this.smoothingWindow) {
      return false;
    }
    
    // Check if top prediction is consistent
    const topClasses = this.predictionHistory.map(preds => preds[0].class);
    const mostCommon = topClasses.reduce((acc, val) => {
      acc[val] = (acc[val] || 0) + 1;
      return acc;
    }, {});
    
    const maxCount = Math.max(...Object.values(mostCommon));
    const stability = maxCount / topClasses.length;
    
    return stability >= threshold;
  }
  
  /**
   * Reset the sequence buffer
   */
  reset() {
    this.sequenceBuffer = [];
    this.predictionHistory = [];
    this.isProcessing = false;
  }
  
  /**
   * Get current statistics
   */
  getStats() {
    return {
      ...this.inferenceStats,
      bufferLength: this.sequenceBuffer.length,
      isReady: this.isReady,
      isProcessing: this.isProcessing,
      historyLength: this.predictionHistory.length
    };
  }
  
  /**
   * Dispose of resources
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
    }
    this.reset();
    this.isReady = false;
    console.log('Inference engine disposed');
  }
}

export default PSLInferenceEngine;
