import React, { useEffect, useRef } from 'react';
import { useWebGLRenderer } from '../../hooks/useWebGLRenderer';
import { useNeuralNetwork } from '../../hooks/useNeuralNetwork';
import './NetworkVisualizer.css';

interface NetworkVisualizerProps {
  networkConfig: {
    layers: number[];
    activations: string[];
  };
  weights?: number[][][];
  activationValues?: number[][];
  gradients?: number[][][];
}

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({
  networkConfig,
  weights,
  activationValues,
  gradients,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { initializeRenderer, renderNetwork } = useWebGLRenderer();

  useEffect(() => {
    if (canvasRef.current) {
      initializeRenderer(canvasRef.current);
    }
  }, [initializeRenderer]);

  useEffect(() => {
    if (weights && activationValues) {
      renderNetwork(networkConfig, weights, activationValues, gradients);
    }
  }, [networkConfig, weights, activationValues, gradients, renderNetwork]);

  return (
    <div className="network-visualizer-container">
      <canvas 
        ref={canvasRef} 
        className="network-canvas"
        width={800}
        height={600}
      />
    </div>
  );
};

export default NetworkVisualizer; 