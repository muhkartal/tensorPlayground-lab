import * as tf from '@tensorflow/tfjs';

/**
 * Creates a Stochastic Gradient Descent (SGD) optimizer.
 * @param learningRate The learning rate to use for the SGD optimizer.
 * @param momentum The momentum parameter.
 */
export const createSGDOptimizer = (
  learningRate: number = 0.01, 
  momentum: number = 0.0
): tf.Optimizer => {
  return tf.train.sgd(learningRate, momentum);
};

/**
 * Creates an Adam optimizer.
 * @param learningRate The learning rate to use for the Adam optimizer.
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon A small constant for numerical stability.
 */
export const createAdamOptimizer = (
  learningRate: number = 0.001, 
  beta1: number = 0.9, 
  beta2: number = 0.999, 
  epsilon: number = 1e-8
): tf.Optimizer => {
  return tf.train.adam(learningRate, beta1, beta2, epsilon);
};

/**
 * Creates an RMSProp optimizer.
 * @param learningRate The learning rate to use for the RMSProp optimizer.
 * @param decay The decay rate.
 * @param momentum The momentum parameter.
 * @param epsilon A small constant for numerical stability.
 * @param centered If true, gradients are normalized by their estimated variance.
 */
export const createRMSPropOptimizer = (
  learningRate: number = 0.001, 
  decay: number = 0.9, 
  momentum: number = 0.0, 
  epsilon: number = 1e-8, 
  centered: boolean = false
): tf.Optimizer => {
  return tf.train.rmsprop(learningRate, decay, momentum, epsilon, centered);
};

/**
 * Creates an AdaGrad optimizer.
 * @param learningRate The learning rate to use for the AdaGrad optimizer.
 * @param initialAccumulatorValue Starting value for the accumulators.
 */
export const createAdaGradOptimizer = (
  learningRate: number = 0.01, 
  initialAccumulatorValue: number = 0.1
): tf.Optimizer => {
  return tf.train.adagrad(learningRate, initialAccumulatorValue);
};

/**
 * Factory function to create an optimizer based on the specified type.
 * @param type The type of optimizer to create.
 * @param learningRate The learning rate to use.
 * @param options Additional optimizer-specific options.
 */
export const createOptimizer = (
  type: string,
  learningRate: number,
  options: Record<string, any> = {}
): tf.Optimizer => {
  switch (type) {
    case 'sgd':
      return createSGDOptimizer(
        learningRate, 
        options.momentum || 0.0
      );
    case 'adam':
      return createAdamOptimizer(
        learningRate, 
        options.beta1 || 0.9, 
        options.beta2 || 0.999, 
        options.epsilon || 1e-8
      );
    case 'rmsprop':
      return createRMSPropOptimizer(
        learningRate, 
        options.decay || 0.9, 
        options.momentum || 0.0, 
        options.epsilon || 1e-8, 
        options.centered || false
      );
    case 'adagrad':
      return createAdaGradOptimizer(
        learningRate, 
        options.initialAccumulatorValue || 0.1
      );
    default:
      console.warn(`Unknown optimizer type: ${type}. Falling back to Adam.`);
      return createAdamOptimizer(learningRate);
  }
}; 