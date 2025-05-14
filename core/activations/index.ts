import * as tf from '@tensorflow/tfjs';

/**
 * Applies ReLU activation: max(0, x)
 */
export const relu = (x: tf.Tensor): tf.Tensor => {
  return tf.relu(x);
};

/**
 * Applies Leaky ReLU activation: x if x > 0, alpha * x otherwise
 */
export const leakyRelu = (x: tf.Tensor, alpha: number = 0.2): tf.Tensor => {
  return tf.leakyRelu(x, alpha);
};

/**
 * Applies Sigmoid activation: 1 / (1 + exp(-x))
 */
export const sigmoid = (x: tf.Tensor): tf.Tensor => {
  return tf.sigmoid(x);
};

/**
 * Applies Tanh activation: tanh(x)
 */
export const tanh = (x: tf.Tensor): tf.Tensor => {
  return tf.tanh(x);
};

/**
 * Applies Softmax activation: exp(x) / sum(exp(x))
 */
export const softmax = (x: tf.Tensor): tf.Tensor => {
  return tf.softmax(x);
};

/**
 * Applies Linear activation (no activation): x
 */
export const linear = (x: tf.Tensor): tf.Tensor => {
  return x;
};

/**
 * Applies SELU (Scaled Exponential Linear Unit) activation
 * SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
 * where alpha and scale are predefined constants
 */
export const selu = (x: tf.Tensor): tf.Tensor => {
  return tf.selu(x);
};

/**
 * Map of activation function names to their implementations
 */
export const activationFunctions = {
  relu,
  leakyRelu,
  sigmoid,
  tanh,
  softmax,
  linear,
  selu,
};

/**
 * Returns the derivative function for the given activation
 */
export const getActivationDerivative = (activationName: string) => {
  switch (activationName) {
    case 'relu':
      return (x: tf.Tensor) => tf.step(x);
    case 'leakyRelu':
      return (x: tf.Tensor) => {
        return tf.where(
          tf.greater(x, 0),
          tf.onesLike(x),
          tf.fill(x.shape, 0.2)
        );
      };
    case 'sigmoid':
      return (x: tf.Tensor) => {
        const sigX = tf.sigmoid(x);
        return tf.mul(sigX, tf.sub(1, sigX));
      };
    case 'tanh':
      return (x: tf.Tensor) => {
        const tanhX = tf.tanh(x);
        return tf.sub(1, tf.square(tanhX));
      };
    case 'linear':
      return (x: tf.Tensor) => tf.onesLike(x);
    case 'selu':
      // Approximate SELU derivative
      return (x: tf.Tensor) => {
        const alpha = 1.6732632423543772;
        const scale = 1.0507009873554805;
        
        return tf.where(
          tf.greater(x, 0),
          tf.fill(x.shape, scale),
          tf.mul(tf.exp(x), tf.fill(x.shape, scale * alpha))
        );
      };
    default:
      return (x: tf.Tensor) => tf.onesLike(x);
  }
}; 