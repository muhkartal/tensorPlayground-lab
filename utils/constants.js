/**
 * Neural Network Playground Constants
 * 
 * This file contains constants used throughout the application.
 */

// Dataset types
export const DATASETS = {
  CIRCLE: 'circle',
  XOR: 'xor',
  GAUSSIAN: 'gaussian',
  SPIRAL: 'spiral',
};

// Activation functions
export const ACTIVATIONS = {
  TANH: 'tanh',
  SIGMOID: 'sigmoid',
  RELU: 'relu',
  LINEAR: 'linear',
};

// Optimizers
export const OPTIMIZERS = {
  SGD: 'sgd',
  ADAM: 'adam',
  RMSPROP: 'rmsprop',
};

// Regularization types
export const REGULARIZATIONS = {
  NONE: 'none',
  L1: 'l1',
  L2: 'l2',
  L1L2: 'l1l2',
};

// Weight initialization strategies
export const WEIGHT_INITIALIZERS = {
  XAVIER: 'xavier',
  HE: 'he',
  RANDOM: 'random',
};

// Problem types
export const PROBLEM_TYPES = {
  CLASSIFICATION: 'classification',
  REGRESSION: 'regression',
};

// UI constants
export const UI = {
  DEFAULT_LAYER_SIZES: [2, 4, 2, 1],
  MIN_NEURONS: 1,
  MAX_NEURONS: 10,
  DEFAULT_LEARNING_RATE: 0.03,
  DEFAULT_MOMENTUM: 0.9,
  DEFAULT_BATCH_SIZE: 10,
}; 