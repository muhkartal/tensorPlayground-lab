/**
 * @typedef {Object} LayerConfig
 * @property {number} units - Number of units in the layer
 * @property {string} activation - Activation function
 * @property {Array<number>} [inputShape] - Input shape for the first layer
 * @property {string} [kernelInitializer] - Weight initialization method
 * @property {string} [biasInitializer] - Bias initialization method
 * @property {Object} [kernelRegularizer] - Regularization settings
 */

/**
 * @typedef {Object} ModelConfig
 * @property {Array<number>} layers - Layer sizes
 * @property {string} activationFunction - Activation function
 * @property {number} learningRate - Learning rate
 * @property {number} momentum - Momentum for SGD optimizer
 * @property {string} optimizer - Optimizer type
 * @property {number} dropoutRate - Dropout rate
 * @property {string} regularizationType - Type of regularization
 * @property {number} regularizationRate - Regularization rate
 * @property {string} weightInitialization - Weight initialization method
 */

/**
 * @typedef {Object} DatasetConfig
 * @property {string} type - Dataset type (circle, xor, gaussian, spiral)
 * @property {number} noiseLevel - Noise level (0-100)
 * @property {number} testRatio - Ratio of test to training data (0-1)
 * @property {boolean} discretizeOutput - Whether to discretize the output
 */

/**
 * @typedef {Object} NeuralNetworkState
 * @property {ModelConfig} modelConfig - Model configuration
 * @property {DatasetConfig} datasetConfig - Dataset configuration
 * @property {boolean} isTraining - Whether the model is currently training
 * @property {number} iterations - Number of training iterations completed
 * @property {number} loss - Current loss value
 * @property {number} accuracy - Current accuracy value
 * @property {Array<number>} lossHistory - History of loss values
 * @property {Array<number>} accuracyHistory - History of accuracy values
 */

// Export the typedefs for JSDoc usage
export {}; 