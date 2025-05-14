import * as tf from '@tensorflow/tfjs';
import { activationFunctions } from '../activations';

/**
 * Dense (fully connected) layer implementation
 */
export class DenseLayer {
  private weights: tf.Tensor2D;
  private bias: tf.Tensor1D;
  private inputShape: number[];
  private outputShape: number[];
  private activation: string;
  private lastInput: tf.Tensor | null = null;
  private lastOutput: tf.Tensor | null = null;
  private lastActivation: tf.Tensor | null = null;

  /**
   * Creates a new dense layer
   * @param inputUnits Number of input units
   * @param outputUnits Number of output units
   * @param activation Activation function to use
   * @param kernelInitializer Weight initialization method
   */
  constructor(
    inputUnits: number,
    outputUnits: number,
    activation: string = 'linear',
    kernelInitializer: string = 'glorotNormal'
  ) {
    this.inputShape = [inputUnits];
    this.outputShape = [outputUnits];
    this.activation = activation;
    
    // Initialize weights and bias
    this.weights = this.initializeWeights(inputUnits, outputUnits, kernelInitializer);
    this.bias = tf.zeros([outputUnits]);
  }
  
  /**
   * Initialize weights using the specified method
   */
  private initializeWeights(
    inputUnits: number,
    outputUnits: number,
    initializer: string
  ): tf.Tensor2D {
    switch (initializer) {
      case 'glorotNormal':
        // Glorot/Xavier normal initialization
        const stddev = Math.sqrt(2 / (inputUnits + outputUnits));
        return tf.randomNormal([inputUnits, outputUnits], 0, stddev);
        
      case 'glorotUniform':
        // Glorot/Xavier uniform initialization
        const limit = Math.sqrt(6 / (inputUnits + outputUnits));
        return tf.randomUniform([inputUnits, outputUnits], -limit, limit);
        
      case 'heNormal':
        // He/Kaiming normal initialization
        const heStddev = Math.sqrt(2 / inputUnits);
        return tf.randomNormal([inputUnits, outputUnits], 0, heStddev);
        
      case 'heUniform':
        // He/Kaiming uniform initialization
        const heLimit = Math.sqrt(6 / inputUnits);
        return tf.randomUniform([inputUnits, outputUnits], -heLimit, heLimit);
        
      case 'zeros':
        return tf.zeros([inputUnits, outputUnits]);
        
      case 'ones':
        return tf.ones([inputUnits, outputUnits]);
        
      default:
        // Default to Glorot normal
        const defaultStddev = Math.sqrt(2 / (inputUnits + outputUnits));
        return tf.randomNormal([inputUnits, outputUnits], 0, defaultStddev);
    }
  }
  
  /**
   * Apply the layer to the input tensor
   * @param input Input tensor
   * @param training Whether the layer is being called during training
   * @returns Output tensor
   */
  forward(input: tf.Tensor, training: boolean = false): tf.Tensor {
    // Save input for backpropagation
    if (training) {
      this.lastInput = input;
    }
    
    // Linear transformation: y = Wx + b
    const linearOutput = tf.add(tf.matMul(input, this.weights), this.bias);
    
    // Apply activation function
    const activationFn = activationFunctions[this.activation] || activationFunctions.linear;
    const output = activationFn(linearOutput);
    
    // Save outputs for backpropagation
    if (training) {
      this.lastOutput = linearOutput;
      this.lastActivation = output;
    }
    
    return output;
  }
  
  /**
   * Compute gradients for backpropagation
   * @param outputGradient Gradient of the loss with respect to the output
   * @returns Gradient of the loss with respect to the input
   */
  backward(outputGradient: tf.Tensor): tf.Tensor {
    if (!this.lastInput || !this.lastOutput) {
      throw new Error('Cannot perform backward pass without a forward pass first');
    }
    
    return tf.tidy(() => {
      // Compute gradient through activation function
      const activationGradient = this.computeActivationGradient(outputGradient);
      
      // Compute gradients with respect to weights and bias
      const weightsGradient = tf.matMul(
        this.lastInput.transpose(), 
        activationGradient
      );
      
      const biasGradient = tf.sum(activationGradient, 0);
      
      // Compute gradient with respect to input
      const inputGradient = tf.matMul(
        activationGradient, 
        this.weights.transpose()
      );
      
      // Return gradients for the previous layer
      return inputGradient;
    });
  }
  
  /**
   * Compute the gradient through the activation function
   */
  private computeActivationGradient(outputGradient: tf.Tensor): tf.Tensor {
    if (!this.lastOutput) {
      throw new Error('Cannot compute activation gradient without a forward pass first');
    }
    
    // For linear activation, the gradient is just the output gradient
    if (this.activation === 'linear') {
      return outputGradient;
    }
    
    // For other activation functions, compute the gradient based on the activation type
    // This is a simplified implementation
    switch (this.activation) {
      case 'relu':
        return tf.mul(outputGradient, tf.step(this.lastOutput));
        
      case 'sigmoid':
        const sigValue = tf.sigmoid(this.lastOutput);
        return tf.mul(outputGradient, tf.mul(sigValue, tf.sub(1, sigValue)));
        
      case 'tanh':
        const tanhValue = tf.tanh(this.lastOutput);
        return tf.mul(outputGradient, tf.sub(1, tf.square(tanhValue)));
        
      default:
        console.warn(`Gradient for ${this.activation} not implemented, using identity`);
        return outputGradient;
    }
  }
  
  /**
   * Update layer parameters with gradients
   * @param weightsGradient Gradient for weights
   * @param biasGradient Gradient for bias
   * @param learningRate Learning rate for update
   */
  updateParameters(
    weightsGradient: tf.Tensor2D, 
    biasGradient: tf.Tensor1D, 
    learningRate: number = 0.01
  ): void {
    // Simple SGD update
    this.weights = tf.sub(
      this.weights, 
      tf.mul(weightsGradient, tf.scalar(learningRate))
    ) as tf.Tensor2D;
    
    this.bias = tf.sub(
      this.bias, 
      tf.mul(biasGradient, tf.scalar(learningRate))
    ) as tf.Tensor1D;
  }
  
  /**
   * Get the output shape of the layer
   */
  getOutputShape(): number[] {
    return this.outputShape;
  }
  
  /**
   * Get the weights of the layer
   */
  getWeights(): tf.Tensor[] {
    return [this.weights, this.bias];
  }
  
  /**
   * Set the weights of the layer
   */
  setWeights(weights: tf.Tensor[]): void {
    if (weights.length !== 2) {
      throw new Error('Expected weights array of length 2');
    }
    
    const [newWeights, newBias] = weights;
    
    // Validate shapes
    if (!this.validateWeightsShape(newWeights as tf.Tensor2D, newBias as tf.Tensor1D)) {
      throw new Error('Invalid weights shape');
    }
    
    this.weights = newWeights as tf.Tensor2D;
    this.bias = newBias as tf.Tensor1D;
  }
  
  /**
   * Validate that the provided weights have the correct shape
   */
  private validateWeightsShape(weights: tf.Tensor2D, bias: tf.Tensor1D): boolean {
    const weightsShape = weights.shape;
    const biasShape = bias.shape;
    
    return (
      weightsShape.length === 2 &&
      weightsShape[0] === this.inputShape[0] &&
      weightsShape[1] === this.outputShape[0] &&
      biasShape.length === 1 &&
      biasShape[0] === this.outputShape[0]
    );
  }
  
  /**
   * Clean up resources
   */
  dispose(): void {
    this.weights.dispose();
    this.bias.dispose();
    
    if (this.lastInput) this.lastInput.dispose();
    if (this.lastOutput) this.lastOutput.dispose();
    if (this.lastActivation) this.lastActivation.dispose();
  }
} 