import * as tf from '@tensorflow/tfjs';

/**
 * Interface for dataset options
 */
interface DatasetOptions {
  numSamples?: number;
  noise?: number;
  seed?: number;
}

/**
 * Class containing methods to generate different datasets for neural network training
 */
export class DatasetGenerator {
  
  /**
   * Generate a circle classification dataset
   * @param options Dataset generation options
   * @returns Object containing input data and labels
   */
  static generateCircleDataset(options: DatasetOptions = {}) {
    const numSamples = options.numSamples || 500;
    const noise = options.noise || 0.1;
    
    // Create points in a circle
    const radius = 5;
    const data: number[][] = [];
    const labels: number[][] = [];
    
    for (let i = 0; i < numSamples; i++) {
      // Generate random angle and radius with some noise
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * radius;
      
      // Add noise to coordinates
      const noiseX = (Math.random() - 0.5) * noise * radius;
      const noiseY = (Math.random() - 0.5) * noise * radius;
      
      const x = r * Math.cos(angle) + noiseX;
      const y = r * Math.sin(angle) + noiseY;
      
      // Determine class based on distance from origin
      const distanceFromOrigin = Math.sqrt(x * x + y * y);
      const isInside = distanceFromOrigin < radius / 2;
      
      data.push([x, y]);
      labels.push(isInside ? [1, 0] : [0, 1]);
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Generate an XOR problem dataset
   * @param options Dataset generation options
   * @returns Object containing input data and labels
   */
  static generateXORDataset(options: DatasetOptions = {}) {
    const numSamples = options.numSamples || 500;
    const noise = options.noise || 0.1;
    
    const data: number[][] = [];
    const labels: number[][] = [];
    
    // Create points for each quadrant
    for (let i = 0; i < numSamples; i++) {
      // Generate random point in [-1, 1] x [-1, 1]
      const x = (Math.random() * 2 - 1) + (Math.random() - 0.5) * noise;
      const y = (Math.random() * 2 - 1) + (Math.random() - 0.5) * noise;
      
      // XOR is true when signs are different
      const xorResult = (x > 0 && y < 0) || (x < 0 && y > 0);
      
      data.push([x, y]);
      labels.push(xorResult ? [1, 0] : [0, 1]);
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Generate a spiral classification dataset
   * @param options Dataset generation options
   * @returns Object containing input data and labels
   */
  static generateSpiralDataset(options: DatasetOptions = {}) {
    const numSamples = options.numSamples || 500;
    const noise = options.noise || 0.1;
    const numClasses = 3; // Number of spiral arms
    
    const data: number[][] = [];
    const labels: number[][] = [];
    
    const samplesPerClass = Math.floor(numSamples / numClasses);
    
    // Generate spirals
    for (let c = 0; c < numClasses; c++) {
      // Create one-hot encoded label
      const label = Array(numClasses).fill(0);
      label[c] = 1;
      
      for (let i = 0; i < samplesPerClass; i++) {
        // Parametric equation for spiral
        const t = 1.0 * i / samplesPerClass * 2 * Math.PI + (2 * Math.PI * c / numClasses);
        const r = 0.2 + 2.0 * t;
        
        // Add noise
        const noiseX = (Math.random() - 0.5) * noise;
        const noiseY = (Math.random() - 0.5) * noise;
        
        // Calculate coordinates
        const x = r * Math.sin(t) + noiseX;
        const y = r * Math.cos(t) + noiseY;
        
        data.push([x, y]);
        labels.push(label);
      }
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Generate a Gaussian clusters dataset
   * @param options Dataset generation options
   * @returns Object containing input data and labels
   */
  static generateGaussianClustersDataset(options: DatasetOptions = {}) {
    const numSamples = options.numSamples || 500;
    const numClusters = 4;
    
    const data: number[][] = [];
    const labels: number[][] = [];
    
    const samplesPerCluster = Math.floor(numSamples / numClusters);
    
    // Cluster centers
    const centers = [
      [-2, -2],
      [-2, 2],
      [2, -2],
      [2, 2]
    ];
    
    // Standard deviation for clusters
    const stdDev = 0.5;
    
    // Generate clusters
    for (let c = 0; c < numClusters; c++) {
      // Create one-hot encoded label
      const label = Array(numClusters).fill(0);
      label[c] = 1;
      
      const [centerX, centerY] = centers[c];
      
      for (let i = 0; i < samplesPerCluster; i++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        
        const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
        
        const x = centerX + z1 * stdDev;
        const y = centerY + z2 * stdDev;
        
        data.push([x, y]);
        labels.push(label);
      }
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Generate a moons dataset (two interleaving half circles)
   * @param options Dataset generation options
   * @returns Object containing input data and labels
   */
  static generateMoonsDataset(options: DatasetOptions = {}) {
    const numSamples = options.numSamples || 500;
    const noise = options.noise || 0.1;
    
    const data: number[][] = [];
    const labels: number[][] = [];
    
    const samplesPerClass = Math.floor(numSamples / 2);
    
    // First moon
    for (let i = 0; i < samplesPerClass; i++) {
      const angle = Math.PI * Math.random();
      
      const x = Math.cos(angle);
      const y = Math.sin(angle);
      
      // Add noise
      const noiseX = (Math.random() - 0.5) * noise;
      const noiseY = (Math.random() - 0.5) * noise;
      
      data.push([x + noiseX, y + noiseY]);
      labels.push([1, 0]);
    }
    
    // Second moon
    for (let i = 0; i < samplesPerClass; i++) {
      const angle = Math.PI * Math.random();
      
      const x = 1 - Math.cos(angle);
      const y = 1 - Math.sin(angle) - 0.5;
      
      // Add noise
      const noiseX = (Math.random() - 0.5) * noise;
      const noiseY = (Math.random() - 0.5) * noise;
      
      data.push([x + noiseX, y + noiseY]);
      labels.push([0, 1]);
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Create a custom dataset from user-provided points
   * @param points Array of points with x, y coordinates and label
   * @returns Object containing input data and labels
   */
  static createCustomDataset(points: Array<{x: number, y: number, label: number}>) {
    if (!points || points.length === 0) {
      throw new Error('No points provided for custom dataset');
    }
    
    const data: number[][] = [];
    const labels: number[][] = [];
    
    // Find the number of unique classes
    const uniqueLabels = new Set(points.map(p => p.label));
    const numClasses = uniqueLabels.size;
    
    for (const point of points) {
      data.push([point.x, point.y]);
      
      // Create one-hot encoded label
      const label = Array(numClasses).fill(0);
      label[point.label] = 1;
      
      labels.push(label);
    }
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(labels)
    };
  }
  
  /**
   * Load dataset from CSV
   * @param csvData CSV data as string
   * @param inputColumns Indices of columns to use as input
   * @param labelColumn Index of column to use as label
   * @param hasHeader Whether the CSV has a header row
   * @returns Object containing input data and labels
   */
  static loadFromCSV(
    csvData: string, 
    inputColumns: number[],
    labelColumn: number,
    hasHeader: boolean = true
  ) {
    // Parse CSV
    const lines = csvData.trim().split('\n');
    const startRow = hasHeader ? 1 : 0;
    
    const data: number[][] = [];
    const labels: Set<number> = new Set();
    const labelMap: Map<string, number> = new Map();
    const rawLabels: string[] = [];
    
    // Process rows
    for (let i = startRow; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const values = line.split(',').map(v => v.trim());
      
      // Extract input features
      const inputValues = inputColumns.map(col => {
        const val = parseFloat(values[col]);
        return isNaN(val) ? 0 : val;
      });
      
      // Extract label
      const labelValue = values[labelColumn];
      rawLabels.push(labelValue);
      
      // Add to data
      data.push(inputValues);
    }
    
    // Convert string labels to numeric indices
    rawLabels.forEach(label => {
      if (!labelMap.has(label)) {
        labelMap.set(label, labelMap.size);
      }
      labels.add(labelMap.get(label)!);
    });
    
    // Create one-hot encoded labels
    const numClasses = labels.size;
    const encodedLabels: number[][] = rawLabels.map(label => {
      const labelIndex = labelMap.get(label)!;
      const encoded = Array(numClasses).fill(0);
      encoded[labelIndex] = 1;
      return encoded;
    });
    
    return {
      xs: tf.tensor2d(data),
      ys: tf.tensor2d(encodedLabels),
      labelMap
    };
  }
} 