import { useCallback, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

interface NetworkConfig {
  layers: number[];
  activations: string[];
  learningRate?: number;
  optimizer?: string;
  regularization?: string;
  regStrength?: number;
}

interface TrainingOptions {
  batchSize: number;
  epochs: number;
  validationSplit?: number;
  callbacks?: tf.CallbackArgs;
}

export const useNeuralNetwork = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [weights, setWeights] = useState<number[][][]>([]);
  const [activations, setActivations] = useState<number[][]>([]);
  const [gradients, setGradients] = useState<number[][][]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState<{ loss: number, accuracy: number }>({ loss: 0, accuracy: 0 });

  // Create and initialize the neural network
  const createNetwork = useCallback((config: NetworkConfig) => {
    // Dispose previous model if exists
    if (model) {
      model.dispose();
    }

    // Map activation function names to TF.js activation functions
    const getActivation = (name: string): string => {
      const activationMap: Record<string, string> = {
        'relu': 'relu',
        'leakyRelu': 'leakyRelu',
        'sigmoid': 'sigmoid',
        'tanh': 'tanh',
        'softmax': 'softmax',
        'linear': 'linear',
        'selu': 'selu',
      };
      return activationMap[name] || 'relu';
    };

    // Create sequential model
    const newModel = tf.sequential();
    
    // Add layers based on configuration
    const { layers, activations } = config;
    
    // Add input layer
    newModel.add(tf.layers.dense({
      units: layers[1],
      inputShape: [layers[0]],
      activation: getActivation(activations[0]),
      kernelInitializer: 'heNormal'
    }));
    
    // Add hidden layers
    for (let i = 1; i < layers.length - 1; i++) {
      newModel.add(tf.layers.dense({
        units: layers[i+1],
        activation: getActivation(activations[i]),
        kernelInitializer: 'heNormal'
      }));
    }
    
    // Configure optimizer
    const lr = config.learningRate || 0.01;
    let optimizer: tf.Optimizer;
    
    switch (config.optimizer) {
      case 'sgd':
        optimizer = tf.train.sgd(lr);
        break;
      case 'adam':
        optimizer = tf.train.adam(lr);
        break;
      case 'rmsprop':
        optimizer = tf.train.rmsprop(lr);
        break;
      case 'adagrad':
        optimizer = tf.train.adagrad(lr);
        break;
      default:
        optimizer = tf.train.adam(lr);
    }
    
    // Configure regularization
    const regConfig: tf.RegularizerIdentifier = {
      identifier: null,
      config: { l1: 0, l2: 0 }
    };
    
    if (config.regularization === 'l1') {
      regConfig.identifier = 'l1l2';
      regConfig.config.l1 = config.regStrength || 0.01;
    } else if (config.regularization === 'l2') {
      regConfig.identifier = 'l1l2';
      regConfig.config.l2 = config.regStrength || 0.01;
    } else if (config.regularization === 'elasticnet') {
      regConfig.identifier = 'l1l2';
      regConfig.config.l1 = (config.regStrength || 0.01) / 2;
      regConfig.config.l2 = (config.regStrength || 0.01) / 2;
    }
    
    // Compile the model
    newModel.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    setModel(newModel);
    
    // Initialize weights, activations and gradients arrays
    extractNetworkState(newModel, config);
    
    return newModel;
  }, [model]);
  
  // Extract weights, activations, and gradients from the model
  const extractNetworkState = useCallback((
    networkModel: tf.LayersModel, 
    config: NetworkConfig
  ) => {
    if (!networkModel) return;
    
    // Extract weights
    const weightsData: number[][][] = [];
    const layerWeights = networkModel.getWeights();
    
    for (let i = 0; i < layerWeights.length; i += 2) {
      const w = layerWeights[i].arraySync() as number[][];
      weightsData.push(w);
    }
    
    setWeights(weightsData);
    
    // Initialize activations with zeros
    const activationsData: number[][] = [];
    for (let i = 0; i < config.layers.length; i++) {
      activationsData.push(new Array(config.layers[i]).fill(0));
    }
    
    setActivations(activationsData);
    
    // Initialize gradients with zeros (same shape as weights)
    const gradientsData: number[][][] = weightsData.map(
      layerWeights => layerWeights.map(
        row => row.map(() => 0)
      )
    );
    
    setGradients(gradientsData);
  }, []);
  
  // Train the neural network
  const trainNetwork = useCallback(async (
    xData: tf.Tensor | number[][], 
    yData: tf.Tensor | number[][], 
    options: TrainingOptions
  ) => {
    if (!model) return;
    
    setIsTraining(true);
    
    const xs = xData instanceof tf.Tensor ? xData : tf.tensor2d(xData as number[][]);
    const ys = yData instanceof tf.Tensor ? yData : tf.tensor2d(yData as number[][]);
    
    try {
      // Custom callback to update weights and activations during training
      const customCallback: tf.CustomCallbackArgs = {
        onBatchEnd: async (_, logs) => {
          // Update metrics
          if (logs) {
            setMetrics({
              loss: logs.loss,
              accuracy: logs.acc || 0
            });
          }
          
          // Get updated weights
          const updatedWeights: number[][][] = [];
          const layerWeights = model.getWeights();
          
          for (let i = 0; i < layerWeights.length; i += 2) {
            const w = layerWeights[i].arraySync() as number[][];
            updatedWeights.push(w);
          }
          
          setWeights(updatedWeights);
          
          // In a real implementation, we would also:
          // 1. Extract activations for visualizing neuron activations
          // 2. Calculate gradients for visualizing flow
          
          // Allow UI to update
          await tf.nextFrame();
        }
      };
      
      // Train the model
      const history = await model.fit(xs, ys, {
        batchSize: options.batchSize,
        epochs: options.epochs,
        validationSplit: options.validationSplit || 0.1,
        callbacks: {
          ...options.callbacks,
          ...customCallback
        }
      });
      
      console.log('Training complete', history);
      
    } catch (error) {
      console.error('Error training network:', error);
    } finally {
      setIsTraining(false);
      
      // Clean up tensors
      if (!(xData instanceof tf.Tensor)) xs.dispose();
      if (!(yData instanceof tf.Tensor)) ys.dispose();
    }
  }, [model]);
  
  // Predict using the neural network
  const predict = useCallback((
    inputData: tf.Tensor | number[][]
  ) => {
    if (!model) return null;
    
    const xs = inputData instanceof tf.Tensor 
      ? inputData 
      : tf.tensor2d(inputData as number[][]);
    
    try {
      // Make prediction
      const prediction = model.predict(xs) as tf.Tensor;
      
      // Update activations for visualization
      // In a real implementation, we would extract intermediate activations
      // from each layer for visualization
      
      return prediction;
    } catch (error) {
      console.error('Error making prediction:', error);
      return null;
    } finally {
      // Clean up tensors
      if (!(inputData instanceof tf.Tensor)) xs.dispose();
    }
  }, [model]);
  
  // Clean up resources on unmount
  useEffect(() => {
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [model]);
  
  return {
    createNetwork,
    trainNetwork,
    predict,
    weights,
    activations,
    gradients,
    isTraining,
    metrics,
  };
}; 