import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';
import {
  Brain,
  Settings,
  Sparkles,
  Play,
  Pause,
  Database,
  Activity,
  Layers,
  Cpu,
  Shuffle,
  Timer,
  TrendingUp,
  Sliders,
  BarChart3,
  Box,
  HelpCircle,
  ChevronRight,
  ChevronDown,
  Info,
  Maximize2,
  Minimize2,
  Copy,
  Download,
  Upload,
  Zap,
  GitBranch,
  Network,
  Eye,
  AlertCircle,
  CheckCircle,
  Loader,
  Terminal,
  Plus,
  Minus,
  X,
} from 'lucide-react';

const NeuralNetworkVisualizer = () => {
  // Core state
  const [layers, setLayers] = useState([2, 4, 2, 1]);
  const [activationFunction, setActivationFunction] = useState('tanh');
  const [learningRate, setLearningRate] = useState(0.03);
  const [momentum, setMomentum] = useState(0.9);
  const [optimizer, setOptimizer] = useState('sgd');
  const [isTraining, setIsTraining] = useState(false);
  const [loss, setLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState('circle');
  const [iterations, setIterations] = useState(0);
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [batchSize, setBatchSize] = useState(10);
  const [lossHistory, setLossHistory] = useState([]);
  const [accuracyHistory, setAccuracyHistory] = useState([]);
  const [dropoutRate, setDropoutRate] = useState(0);
  const [regularizationType, setRegularizationType] = useState('none');
  const [regularizationRate, setRegularizationRate] = useState(0);
  const [weightInitialization, setWeightInitialization] = useState('xavier');
  const [trainingSpeed, setTrainingSpeed] = useState(1);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [testRatio, setTestRatio] = useState(0.1);
  const [discretizeOutput, setDiscretizeOutput] = useState(false);
  const [showTestData, setShowTestData] = useState(false);
  const [problemType, setProblemType] = useState('classification');
  const [dataCache, setDataCache] = useState(null);
  const [trainDataCache, setTrainDataCache] = useState(null);
  const [testDataCache, setTestDataCache] = useState(null);

  // UI state
  const [activeTab, setActiveTab] = useState('architecture');
  const [showHelp, setShowHelp] = useState(false);
  const [expandedSection, setExpandedSection] = useState('layers');
  const [consoleMessages, setConsoleMessages] = useState([]);

  // Refs
  const svgRef = useRef(null);
  const chartRef = useRef(null);
  const dataVisRef = useRef(null); // Add reference for data visualization
  const modelRef = useRef(null);
  const trainingControlRef = useRef(false);
  const animationRef = useRef(null);

  // Store activations for visualization
  const [activations, setActivations] = useState([]);
  const [weights, setWeights] = useState([]);

  // Update network weights from model
  const updateWeights = () => {
    if (!modelRef.current) return;

    try {
      const newWeights = [];

      // Extract weights from model
      for (let i = 0; i < layers.length - 1; i++) {
        const layerWeights = modelRef.current.layers[i].getWeights();

        if (!layerWeights || layerWeights.length === 0) {
          newWeights.push(null);
          continue;
        }

        // Get the weight matrix (not the bias)
        const weightMatrix = layerWeights[0];
        const values = weightMatrix.arraySync();

        // Add to our weight array
        newWeights.push(values);
      }

      setWeights(newWeights);
    } catch (error) {
      console.error('Error updating weights:', error);

      // Create some initial random weights if needed
      const fakeWeights = [];
      for (let i = 0; i < layers.length - 1; i++) {
        const inputSize = layers[i];
        const outputSize = layers[i + 1];

        const layerWeights = [];
        for (let j = 0; j < inputSize; j++) {
          const neuronWeights = [];
          for (let k = 0; k < outputSize; k++) {
            // Random small initial weights
            neuronWeights.push((Math.random() * 2 - 1) * 0.5);
          }
          layerWeights.push(neuronWeights);
        }
        fakeWeights.push(layerWeights);
      }

      setWeights(fakeWeights);
    }
  };

  // Update the loss/accuracy chart
  const updateChart = (lossHistory, accuracyHistory) => {
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();

    if (lossHistory.length === 0) return;

    const width = chartRef.current.clientWidth || 600;
    const height = chartRef.current.clientHeight || 200;
    const margin = { top: 20, right: 40, bottom: 30, left: 50 };

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Set background
    svg.append('rect').attr('width', width).attr('height', height).attr('fill', '#0a0a0b');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Create scales
    const x = d3
      .scaleLinear()
      .domain([0, lossHistory.length - 1])
      .range([0, chartWidth]);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(lossHistory) * 1.1 || 1])
      .range([chartHeight, 0]);

    // Create axes
    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(x).ticks(5))
      .attr('color', 'rgba(255, 255, 255, 0.3)')
      .attr('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .attr('color', 'rgba(255, 255, 255, 0.3)')
      .attr('font-size', '10px');

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(x).ticks(10).tickSize(-chartHeight).tickFormat(''))
      .attr('color', 'rgba(255, 255, 255, 0.05)');

    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(5).tickSize(-chartWidth).tickFormat(''))
      .attr('color', 'rgba(255, 255, 255, 0.05)');

    // Create line generator
    const line = d3
      .line()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);

    // Add loss line
    g.append('path')
      .datum(lossHistory)
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add accuracy line if present
    if (accuracyHistory && accuracyHistory.length > 0) {
      const y2 = d3.scaleLinear().domain([0, 1]).range([chartHeight, 0]);

      const accuracyLine = d3
        .line()
        .x((d, i) => x(i))
        .y(d => y2(d))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(accuracyHistory)
        .attr('fill', 'none')
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2)
        .attr('d', accuracyLine);

      g.append('g')
        .attr('transform', `translate(${chartWidth},0)`)
        .call(d3.axisRight(y2).ticks(5))
        .attr('color', 'rgba(255, 255, 255, 0.3)')
        .attr('font-size', '10px');
    }

    // Add legends
    const legend = svg
      .append('g')
      .attr('transform', `translate(${margin.left + 10}, ${margin.top + 10})`);

    legend.append('rect').attr('x', 0).attr('width', 12).attr('height', 12).attr('fill', '#ef4444');

    legend
      .append('text')
      .attr('x', 20)
      .attr('y', 10)
      .attr('font-size', '12px')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .text('Loss');

    if (accuracyHistory && accuracyHistory.length > 0) {
      legend
        .append('rect')
        .attr('x', 70)
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', '#3b82f6');

      legend
        .append('text')
        .attr('x', 90)
        .attr('y', 10)
        .attr('font-size', '12px')
        .attr('fill', 'rgba(255, 255, 255, 0.7)')
        .text('Accuracy');
    }
  };

  // Update layer size when changed in UI
  const updateLayerSize = (layerIndex, newSize) => {
    if (isTraining) return;

    const size = parseInt(newSize);
    if (isNaN(size) || size < 1) return;

    // Update layer size
    const newLayers = [...layers];
    newLayers[layerIndex] = size;
    setLayers(newLayers);

    // Reset model
    if (modelRef.current) {
      modelRef.current = null;
    }
  };

  // Add a new hidden layer
  const addLayer = () => {
    if (isTraining) return;

    // Add a new hidden layer before the output layer
    const newLayers = [...layers];
    const outputLayerSize = newLayers.pop(); // Remove output layer

    // Add new hidden layer with average size of previous hidden layer
    const prevHiddenLayerSize = newLayers[newLayers.length - 1];
    newLayers.push(prevHiddenLayerSize);

    // Add back output layer
    newLayers.push(outputLayerSize);

    setLayers(newLayers);

    // Reset model
    if (modelRef.current) {
      modelRef.current = null;
    }
  };

  // Remove a hidden layer
  const removeLayer = () => {
    if (isTraining) return;

    // Only remove if we have more than 2 layers (input + output)
    if (layers.length <= 2) return;

    // Remove second-to-last layer (the last hidden layer)
    const newLayers = [...layers];
    const outputLayerSize = newLayers.pop(); // Remove output layer
    newLayers.pop(); // Remove last hidden layer
    newLayers.push(outputLayerSize); // Add back output layer

    setLayers(newLayers);

    // Reset model
    if (modelRef.current) {
      modelRef.current = null;
    }
  };

  // Dark theme styles (keeping same UI)
  const styles = {
    container: {
      minHeight: '100vh',
      background: '#0a0a0b',
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      color: '#e5e7eb',
      overflow: 'hidden',
    },

    header: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 50,
      height: '60px',
      background: 'rgba(10, 10, 11, 0.95)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 24px',
      backdropFilter: 'blur(8px)',
    },

    headerContent: {
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },

    logo: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
    },

    logoText: {
      fontSize: '20px',
      fontWeight: '600',
      color: '#f9fafb',
    },

    iconButton: {
      width: '36px',
      height: '36px',
      borderRadius: '8px',
      background: 'rgba(255, 255, 255, 0.05)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#9ca3af',
      cursor: 'pointer',
      transition: 'all 0.2s',
      ':hover': {
        background: 'rgba(255, 255, 255, 0.1)',
        color: '#e5e7eb',
      },
    },

    mainContent: {
      paddingTop: '60px',
      display: 'grid',
      gridTemplateColumns: '320px 1fr',
      height: '100vh',
      gap: 0,
    },

    sidebar: {
      borderRight: '1px solid rgba(255, 255, 255, 0.08)',
      background: '#111113',
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 60px)',
    },

    tabs: {
      display: 'flex',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      padding: '0 16px',
    },

    tab: {
      flex: 1,
      height: '48px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      cursor: 'pointer',
      fontSize: '14px',
      color: '#6b7280',
      borderBottom: '2px solid transparent',
      transition: 'all 0.2s',
    },

    activeTab: {
      color: '#f9fafb',
      borderBottomColor: '#3b82f6',
    },

    tabContent: {
      flex: 1,
      overflow: 'auto',
      padding: '16px',
      minHeight: 0,
    },

    section: {
      marginBottom: '16px',
    },

    sectionHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '10px 14px',
      background: 'rgba(13, 13, 20, 0.6)',
      borderRadius: '10px',
      cursor: 'pointer',
      marginBottom: '12px',
      fontSize: '14px',
      fontWeight: '500',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.05)',
    },

    sectionTitle: {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      color: '#e5e7eb',
    },

    sectionContent: {
      padding: '12px 14px',
      display: 'grid',
      gap: '16px',
      background: 'rgba(13, 13, 20, 0.3)',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.03)',
    },

    formGroup: {
      display: 'grid',
      gap: '8px',
    },

    label: {
      fontSize: '13px',
      color: '#9ca3af',
      fontWeight: '500',
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
    },

    input: {
      width: '100%',
      height: '36px',
      padding: '0 12px',
      background: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '6px',
      color: '#e5e7eb',
      fontSize: '14px',
      outline: 'none',
      transition: 'all 0.2s',
    },

    select: {
      width: '100%',
      height: '36px',
      padding: '0 12px',
      background: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '6px',
      color: '#e5e7eb',
      fontSize: '14px',
      cursor: 'pointer',
      outline: 'none',
    },

    slider: {
      WebkitAppearance: 'none',
      width: '100%',
      height: '6px',
      background: 'rgba(13, 13, 20, 0.6)',
      borderRadius: '3px',
      outline: 'none',
      cursor: 'pointer',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.2)',
    },

    sliderValue: {
      fontSize: '13px',
      color: '#60a5fa',
      fontWeight: '600',
      marginLeft: '8px',
    },

    layerGrid: {
      display: 'flex',
      gap: '8px',
      marginTop: '8px',
      flexWrap: 'wrap',
      alignItems: 'center',
      justifyContent: 'center',
    },

    layerInputWrapper: {
      flex: '0 0 auto',
      minWidth: '60px',
      position: 'relative',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    },

    layerInput: {
      width: '100%',
      height: '46px',
      background: 'rgba(13, 13, 20, 0.6)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '8px',
      color: '#e5e7eb',
      textAlign: 'center',
      fontSize: '18px',
      fontWeight: '600',
      outline: 'none',
      transition: 'all 0.2s',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
    },

    layerLabel: {
      fontSize: '12px',
      color: '#9ca3af',
      marginTop: '6px',
      textAlign: 'center',
      width: '100%',
      fontWeight: '500',
    },

    layerArrow: {
      color: '#6b7280',
      fontSize: '16px',
      alignSelf: 'center',
      padding: '0 4px',
      marginTop: '-4px',
    },

    buttonGroup: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '12px',
    },

    button: {
      height: '38px',
      borderRadius: '10px',
      border: 'none',
      cursor: 'pointer',
      fontSize: '14px',
      fontWeight: '500',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      transition: 'all 0.2s',
    },

    primaryButton: {
      background: 'linear-gradient(to right, #3b82f6, #4f46e5)',
      color: 'white',
      boxShadow: '0 2px 6px rgba(59, 130, 246, 0.4)',
    },

    secondaryButton: {
      background: 'rgba(13, 13, 20, 0.6)',
      color: '#d1d5db',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    },

    trainingControls: {
      padding: '16px',
      borderTop: '1px solid rgba(255, 255, 255, 0.08)',
      background: 'rgba(0, 0, 0, 0.3)',
      marginTop: 'auto',
    },

    trainButton: {
      width: '100%',
      height: '44px',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      cursor: 'pointer',
      fontSize: '15px',
      fontWeight: '500',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      transition: 'all 0.2s',
      background: 'rgba(255, 255, 255, 0.05)',
      backdropFilter: 'blur(8px)',
      color: '#e5e7eb',
    },

    visualizationArea: {
      background: '#0a0a0b',
      display: 'flex',
      flexDirection: 'column',
    },

    metricsBar: {
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '1px',
      background: 'rgba(255, 255, 255, 0.08)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    },

    metricCard: {
      padding: '16px',
      background: '#0a0a0b',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
    },

    metricLabel: {
      fontSize: '12px',
      color: '#6b7280',
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
    },

    metricValue: {
      fontSize: '24px',
      fontWeight: '600',
      color: '#f9fafb',
    },

    metricChange: {
      fontSize: '12px',
      display: 'flex',
      alignItems: 'center',
      gap: '4px',
    },

    metricPositive: {
      color: '#22c55e',
    },

    metricNegative: {
      color: '#ef4444',
    },

    visualizationPanels: {
      flex: 1,
      overflow: 'hidden',
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '16px',
      margin: '16px 0',
      height: '350px',
    },

    dataPanel: {
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '8px',
      background: '#111113',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    },

    dataContainer: {
      flex: 1,
      overflow: 'hidden',
    },

    dataSvg: {
      width: '100%',
      height: '100%',
    },

    checkboxGroup: {
      display: 'flex',
      alignItems: 'center',
      gap: '4px',
    },

    checkbox: {
      cursor: 'pointer',
    },

    checkboxLabel: {
      fontSize: '12px',
      color: '#9ca3af',
      cursor: 'pointer',
    },

    networkPanel: {
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      flexDirection: 'column',
    },

    panelHeader: {
      padding: '12px 16px',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      background: 'rgba(0, 0, 0, 0.2)',
    },

    panelTitle: {
      fontSize: '14px',
      fontWeight: '500',
      color: '#e5e7eb',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },

    networkContainer: {
      flex: 1,
      overflow: 'hidden',
      minHeight: '400px',
      position: 'relative', // Add position relative
    },

    networkSvg: {
      width: '100%',
      height: '100%',
      position: 'absolute', // Make SVG position absolute
      top: 0,
      left: 0,
    },

    chartPanel: {
      height: '250px',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      flexDirection: 'column',
    },

    chartContainer: {
      flex: 1,
      overflow: 'hidden',
    },

    chart: {
      width: '100%',
      height: '100%',
    },

    console: {
      background: '#0a0a0b',
      borderTop: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      flexDirection: 'column',
      maxHeight: '160px',
    },

    consoleHeader: {
      padding: '8px 16px',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      fontSize: '13px',
      color: '#9ca3af',
      background: 'rgba(0, 0, 0, 0.2)',
    },

    consoleContent: {
      padding: '8px 16px',
      fontFamily: 'Monaco, Consolas, "Ubuntu Mono", monospace',
      fontSize: '12px',
      color: '#e5e7eb',
      overflow: 'auto',
      flex: 1,
      minHeight: '100px',
    },

    consoleLine: {
      marginBottom: '2px',
      lineHeight: 1.5,
    },

    consoleTimestamp: {
      color: '#6b7280',
    },

    helpOverlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      zIndex: 100,
      display: showHelp ? 'block' : 'none',
    },

    helpPanel: {
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: '90%',
      maxWidth: '600px',
      background: '#111113',
      borderRadius: '12px',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      padding: '24px',
      maxHeight: '80vh',
      overflow: 'auto',
    },

    helpHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '20px',
    },

    helpTitle: {
      fontSize: '20px',
      fontWeight: '600',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },

    helpContent: {
      display: 'grid',
      gap: '16px',
    },

    helpSection: {
      display: 'grid',
      gap: '8px',
    },

    helpSectionTitle: {
      fontSize: '16px',
      fontWeight: '500',
      color: '#e5e7eb',
    },

    helpText: {
      fontSize: '14px',
      color: '#9ca3af',
      lineHeight: 1.6,
    },
  };

  // Dataset generators
  const generateDataset = async type => {
    try {
      let data = [];
      let labels = [];

      const addNoise = (x, y, noiseAmount) => {
        if (noiseAmount === 0) return [x, y];

        // Add scaled noise based on noise level (0-50)
        const noise = noiseAmount / 25; // Scale to 0-2 range
        return [x + (Math.random() * 2 - 1) * noise, y + (Math.random() * 2 - 1) * noise];
      };

      switch (type) {
        case 'xor':
          // Add the classic XOR cases with potential noise
          for (let i = 0; i < 100; i++) {
            let [x1, y1] = addNoise(0, 0, noiseLevel);
            data.push([x1, y1]);
            labels.push([0]);

            let [x2, y2] = addNoise(0, 1, noiseLevel);
            data.push([x2, y2]);
            labels.push([1]);

            let [x3, y3] = addNoise(1, 0, noiseLevel);
            data.push([x3, y3]);
            labels.push([1]);

            let [x4, y4] = addNoise(1, 1, noiseLevel);
            data.push([x4, y4]);
            labels.push([0]);
          }
          break;
        case 'circle':
          for (let i = 0; i < 300; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.random() * 6; // Scale up to match TensorFlow Playground
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            // Add noise
            const [nx, ny] = addNoise(x, y, noiseLevel);
            data.push([nx, ny]);

            // Circle classification - points inside radius 3 are class 1
            const originalDistance = Math.sqrt(x * x + y * y);
            labels.push([originalDistance < 3 ? 1 : 0]);
          }
          break;
        case 'spiral':
          const turns = 2; // Number of spiral turns
          for (let i = 0; i < 200; i++) {
            // Create two classes of spirals
            for (let label = 0; label < 2; label++) {
              const r = (i / 200) * 6; // Scale up for TensorFlow Playground
              const theta = r * turns * Math.PI * 2 + label * Math.PI;

              const x = r * Math.cos(theta);
              const y = r * Math.sin(theta);

              // Add noise
              const [nx, ny] = addNoise(x, y, noiseLevel);
              data.push([nx, ny]);
              labels.push([label]);
            }
          }
          break;
        case 'gaussian':
          // Create two Gaussian clusters
          for (let i = 0; i < 200; i++) {
            // Class 0: centered at (-2, -2)
            const x1 = -2 + Math.random() * 2 - 1;
            const y1 = -2 + Math.random() * 2 - 1;

            // Add noise
            const [nx1, ny1] = addNoise(x1, y1, noiseLevel);
            data.push([nx1, ny1]);
            labels.push([0]);

            // Class 1: centered at (2, 2)
            const x2 = 2 + Math.random() * 2 - 1;
            const y2 = 2 + Math.random() * 2 - 1;

            // Add noise
            const [nx2, ny2] = addNoise(x2, y2, noiseLevel);
            data.push([nx2, ny2]);
            labels.push([1]);
          }
          break;
        default:
          // Default to circle dataset
          for (let i = 0; i < 300; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = Math.random() * 6;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            const [nx, ny] = addNoise(x, y, noiseLevel);
            data.push([nx, ny]);

            const originalDistance = Math.sqrt(x * x + y * y);
            labels.push([originalDistance < 3 ? 1 : 0]);
          }
      }

      // Convert to tensors
      const xs = tf.tensor2d(data);
      const ys = tf.tensor2d(labels);

      // Split data using a simpler approach if the tf.gather method fails
      let trainData, testData;

      try {
        // Try the tensor-based split
        const split = splitTrainTest({ xs, ys }, testRatio);
        trainData = split.train;
        testData = split.test;
      } catch (error) {
        // Fallback to a simple manual split
        console.warn('Using fallback data splitting method:', error);

        const splitIdx = Math.floor(data.length * (1 - testRatio));
        const trainXs = tf.tensor2d(data.slice(0, splitIdx));
        const trainYs = tf.tensor2d(labels.slice(0, splitIdx));
        const testXs = tf.tensor2d(data.slice(splitIdx));
        const testYs = tf.tensor2d(labels.slice(splitIdx));

        trainData = { xs: trainXs, ys: trainYs };
        testData = { xs: testXs, ys: testYs };
      }

      // Store data in state
      setDataCache({ xs, ys });
      setTrainDataCache(trainData);
      setTestDataCache(testData);

      console.log(`Generated ${type} dataset with ${data.length} samples`);
      setConsoleMessages(prev => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Generated ${type} dataset with ${data.length} samples (${Math.round(
            (1 - testRatio) * 100
          )}% train, ${Math.round(testRatio * 100)}% test)`,
          type: 'info',
        },
      ]);

      // Update visualizations
      visualizeDataset();

      return { xs, ys };
    } catch (error) {
      console.error('Error generating dataset:', error);
      setConsoleMessages(prev => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Error generating dataset: ${error.message}`,
          type: 'error',
        },
      ]);

      // Return fallback empty dataset
      return {
        xs: tf.tensor2d([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]),
        ys: tf.tensor2d([[0], [1], [1], [0]]),
      };
    }
  };

  // Create model with TensorFlow.js
  const createModel = () => {
    const model = tf.sequential();

    const getInitializer = () => {
      switch (weightInitialization) {
        case 'xavier':
          return 'glorotNormal';
        case 'he':
          return 'heNormal';
        case 'random':
          return 'randomNormal';
        default:
          return 'glorotNormal';
      }
    };

    // Add all layers
    for (let i = 0; i < layers.length - 1; i++) {
      model.add(
        tf.layers.dense({
          units: layers[i + 1],
          activation:
            i === layers.length - 2 && selectedDataset !== 'sine' ? 'sigmoid' : activationFunction,
          inputShape: i === 0 ? [layers[i]] : undefined,
          kernelInitializer: getInitializer(),
          biasInitializer: 'zeros',
          kernelRegularizer:
            regularizationType !== 'none' ? tf.regularizers.l2({ l2: regularizationRate }) : null,
        })
      );

      if (dropoutRate > 0 && i < layers.length - 2) {
        model.add(tf.layers.dropout({ rate: dropoutRate }));
      }
    }

    // Create optimizer
    let optimizerInstance;
    switch (optimizer) {
      case 'adam':
        optimizerInstance = tf.train.adam(learningRate);
        break;
      case 'rmsprop':
        optimizerInstance = tf.train.rmsprop(learningRate);
        break;
      case 'sgd':
        optimizerInstance = tf.train.sgd(learningRate);
        if (momentum > 0) {
          optimizerInstance = tf.train.momentum(learningRate, momentum);
        }
        break;
      default:
        optimizerInstance = tf.train.sgd(learningRate);
    }

    // Compile model
    model.compile({
      optimizer: optimizerInstance,
      loss: selectedDataset === 'sine' ? 'meanSquaredError' : 'binaryCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
  };

  // Network visualization with activations
  const visualizeNetwork = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const rect = svgRef.current.getBoundingClientRect();
    const width = rect.width || 900;
    const height = rect.height || 500;
    const margin = 60;

    svg.attr('width', width).attr('height', height);

    // Set background
    svg.append('rect').attr('width', width).attr('height', height).attr('fill', '#0c0c10');

    // Add subtle grid pattern
    const defs = svg.append('defs');
    const gridPattern = defs
      .append('pattern')
      .attr('id', 'grid')
      .attr('width', 20)
      .attr('height', 20)
      .attr('patternUnits', 'userSpaceOnUse');

    gridPattern
      .append('path')
      .attr('d', 'M 20 0 L 0 0 0 20')
      .attr('fill', 'none')
      .attr('stroke', 'rgba(255, 255, 255, 0.03)')
      .attr('stroke-width', 1);

    svg.append('rect').attr('width', width).attr('height', height).attr('fill', 'url(#grid)');

    // Add a glow filter for neurons
    const glowFilter = defs
      .append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    glowFilter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');

    glowFilter
      .append('feComposite')
      .attr('in', 'SourceGraphic')
      .attr('in2', 'blur')
      .attr('operator', 'over');

    // Calculate positions
    const layerWidth = (width - 2 * margin) / (layers.length - 1);

    // Initialize weights if needed
    if (weights.length === 0) {
      const newWeights = [];
      for (let i = 0; i < layers.length - 1; i++) {
        const layerWeights = [];
        for (let j = 0; j < layers[i]; j++) {
          const neuronWeights = [];
          for (let k = 0; k < layers[i + 1]; k++) {
            // Initialize with small random values instead of zeros for better visualization
            neuronWeights.push(Math.random() * 0.5 - 0.25);
          }
          layerWeights.push(neuronWeights);
        }
        newWeights.push(layerWeights);
      }
      setWeights(newWeights);
    }

    // Initialize activations if needed
    if (activations.length === 0) {
      const newActivations = [];
      for (let i = 0; i < layers.length; i++) {
        const layerActivations = [];
        for (let j = 0; j < layers[i]; j++) {
          // Small random activations for better initial visualization
          layerActivations.push(Math.random() * 0.5 - 0.25);
        }
        newActivations.push(layerActivations);
      }
      setActivations(newActivations);
    }

    // Find max weight for better color scaling
    let maxWeight = 0.1; // Set minimum to avoid division by zero
    for (let i = 0; i < weights.length; i++) {
      if (!weights[i]) continue;
      for (let j = 0; j < weights[i].length; j++) {
        if (!weights[i][j]) continue;
        for (let k = 0; k < weights[i][j].length; k++) {
          if (weights[i][j][k] !== undefined) {
            maxWeight = Math.max(maxWeight, Math.abs(weights[i][j][k]));
          }
        }
      }
    }

    // Calculate neuron positions for all layers upfront
    const neuronPositions = [];
    for (let i = 0; i < layers.length; i++) {
      const layerPositions = [];
      // Calculate vertical spacing based on the number of neurons in this layer
      const verticalSpacing = Math.min((height - 2 * margin) / layers[i], 70);
      const layerHeight = verticalSpacing * layers[i];
      const startY = (height - layerHeight) / 2;

      for (let j = 0; j < layers[i]; j++) {
        const x = margin + i * layerWidth;
        const y = startY + j * verticalSpacing + verticalSpacing / 2;
        layerPositions.push({ x, y });
      }
      neuronPositions.push(layerPositions);
    }

    // Layer labels with improved aesthetics
    for (let i = 0; i < layers.length; i++) {
      const x = margin + i * layerWidth;
      let label =
        i === 0
          ? `Input (${layers[i]})`
          : i === layers.length - 1
          ? `Output (${layers[i]})`
          : `Hidden ${i} (${layers[i]})`;

      const labelGroup = svg.append('g').attr('transform', `translate(${x}, 30)`);

      // Improved label background
      labelGroup
        .append('rect')
        .attr('x', -55)
        .attr('y', -14)
        .attr('width', 110)
        .attr('height', 28)
        .attr('rx', 14)
        .attr('fill', '#1c1c24')
        .attr('stroke', 'rgba(255, 255, 255, 0.1)')
        .attr('stroke-width', 1)
        .attr('filter', 'drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.2))');

      labelGroup
        .append('text')
        .attr('x', 0)
        .attr('y', 4)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', '600')
        .attr('fill', '#c4cfd9')
        .text(label);
    }

    // Find max activation for better color scaling
    let maxActivation = 0.1; // Set minimum to avoid division by zero
    for (let i = 0; i < activations.length; i++) {
      if (!activations[i]) continue;
      for (let j = 0; j < activations[i].length; j++) {
        if (activations[i][j] !== undefined) {
          maxActivation = Math.max(maxActivation, Math.abs(activations[i][j]));
        }
      }
    }

    // Create a group for connections first (so they appear under neurons)
    const connectionsGroup = svg.append('g').attr('class', 'connections');

    // DRAW CONNECTIONS BETWEEN NEURONS
    // Draw all connections as straight lines first - guaranteed to be visible
    for (let i = 0; i < layers.length - 1; i++) {
      const fromLayer = i;
      const toLayer = i + 1;

      // Draw connections from each neuron in this layer to each neuron in the next layer
      for (let fromNeuron = 0; fromNeuron < layers[fromLayer]; fromNeuron++) {
        for (let toNeuron = 0; toNeuron < layers[toLayer]; toNeuron++) {
          // Get positions of the connected neurons
          const fromPos = neuronPositions[fromLayer][fromNeuron];
          const toPos = neuronPositions[toLayer][toNeuron];

          // Get the weight value if available
          let weight = 0;
          try {
            if (
              weights[fromLayer] &&
              weights[fromLayer][fromNeuron] &&
              weights[fromLayer][fromNeuron][toNeuron] !== undefined
            ) {
              weight = weights[fromLayer][fromNeuron][toNeuron];
            } else {
              // Use small random weight for visual purposes if not defined
              weight = Math.random() * 0.5 - 0.25;
            }
          } catch (error) {
            console.error('Error accessing weight:', error);
            weight = Math.random() * 0.5 - 0.25;
          }

          // Normalize weight for visualization
          const normalizedWeight = Math.abs(weight) / maxWeight;

          // Set connection style based on weight
          const strokeWidth = Math.max(0.5, Math.min(3, 0.5 + normalizedWeight * 2.5));
          const strokeOpacity = Math.max(0.2, Math.min(0.9, 0.2 + normalizedWeight * 0.7));

          // Set color based on weight sign
          const strokeColor =
            weight >= 0
              ? `rgba(130, 195, 236, ${Math.min(0.95, 0.3 + normalizedWeight * 0.65)})`
              : `rgba(231, 137, 137, ${Math.min(0.95, 0.3 + normalizedWeight * 0.65)})`;

          // Create a unique connection ID
          const connectionId = `conn-${fromLayer}-${fromNeuron}-${toLayer}-${toNeuron}`;

          // Add the connection line with proper styling
          connectionsGroup
            .append('line')
            .attr('id', connectionId)
            .attr(
              'class',
              `connection layer-${fromLayer}-${fromNeuron} layer-${toLayer}-${toNeuron}`
            )
            .attr('x1', fromPos.x)
            .attr('y1', fromPos.y)
            .attr('x2', toPos.x)
            .attr('y2', toPos.y)
            .attr('stroke', strokeColor)
            .attr('stroke-width', strokeWidth)
            .attr('stroke-opacity', strokeOpacity)
            .attr('data-weight', weight)
            .attr('data-from-layer', fromLayer)
            .attr('data-from-neuron', fromNeuron)
            .attr('data-to-layer', toLayer)
            .attr('data-to-neuron', toNeuron);
        }
      }
    }

    // Draw neurons with activations
    for (let i = 0; i < layers.length; i++) {
      const layerActivations = activations[i] || Array(layers[i]).fill(0);

      // Create a group for this layer's neurons
      const layerGroup = svg.append('g').attr('class', `layer-${i}`);

      for (let j = 0; j < layers[i]; j++) {
        const position = neuronPositions[i][j];

        const neuronGroup = layerGroup
          .append('g')
          .attr('class', 'neuron-group')
          .attr('transform', `translate(${position.x}, ${position.y})`);

        // Get activation value
        const activation = layerActivations[j] || 0;
        const normalizedActivation = Math.abs(activation) / maxActivation;

        // Add subtle glow effect for active neurons
        if (normalizedActivation > 0.2) {
          neuronGroup
            .append('circle')
            .attr('r', 24)
            .attr('fill', activation > 0 ? 'rgba(82, 156, 204, 0.2)' : 'rgba(204, 82, 82, 0.2)')
            .attr('filter', 'url(#glow)');
        }

        // Calculate fill color based on activation
        const fillColor =
          activation > 0
            ? `rgba(130, 195, 236, ${Math.min(0.95, 0.3 + normalizedActivation * 0.65)})`
            : `rgba(231, 137, 137, ${Math.min(0.95, 0.3 + normalizedActivation * 0.65)})`;

        const edgeColor = activation > 0 ? '#9ecdf9' : '#f9c09e';

        // Draw the neuron
        neuronGroup
          .append('circle')
          .attr('class', `neuron layer-${i} neuron-${j}`)
          .attr('r', 20)
          .attr('fill', fillColor)
          .attr('stroke', edgeColor)
          .attr('stroke-width', 2)
          .attr('filter', 'drop-shadow(0px 2px 3px rgba(0, 0, 0, 0.25))')
          .style('cursor', 'pointer')
          .on('click', function () {
            setSelectedNeuron({ layer: i, neuron: j });
            highlightNeuron(i, j);
          });

        // Add a subtle inner circle for depth
        neuronGroup
          .append('circle')
          .attr('r', 14)
          .attr('fill', 'none')
          .attr('stroke', 'rgba(255, 255, 255, 0.1)')
          .attr('stroke-width', 1);

        // Value display with improved typography
        neuronGroup
          .append('text')
          .attr('class', `neuron-value layer-${i}-neuron-${j}-value`)
          .attr('text-anchor', 'middle')
          .attr('dy', '0.35em')
          .attr('font-size', 12)
          .attr('font-weight', '600')
          .attr('fill', '#ffffff')
          .attr('filter', 'drop-shadow(0px 1px 1px rgba(0, 0, 0, 0.8))')
          .text(activation !== undefined ? activation.toFixed(3) : '0.000');
      }
    }

    // Add network state information
    svg
      .append('text')
      .attr('x', width - 20)
      .attr('y', 20)
      .attr('text-anchor', 'end')
      .attr('font-size', 12)
      .attr('fill', 'rgba(255, 255, 255, 0.5)')
      .text(`Iteration: ${iterations}`);

    if (lossHistory.length > 0) {
      svg
        .append('text')
        .attr('x', width - 20)
        .attr('y', 40)
        .attr('text-anchor', 'end')
        .attr('font-size', 12)
        .attr('fill', 'rgba(255, 255, 255, 0.5)')
        .text(`Loss: ${lossHistory[lossHistory.length - 1].toFixed(4)}`);
    }
  };

  // Highlight selected neuron
  const highlightNeuron = (layer, neuron) => {
    const svg = d3.select(svgRef.current);

    // Reset all neurons to their normal state
    svg
      .selectAll('.neuron')
      .attr('r', 20)
      .attr('fill', function () {
        // Extract layer and neuron info from class
        const classes = d3.select(this).attr('class');
        const match = classes.match(/layer-(\d+) neuron-(\d+)/);
        if (match) {
          const l = parseInt(match[1]);
          const n = parseInt(match[2]);

          // Get the activation
          if (activations[l] && activations[l][n] !== undefined) {
            const activation = activations[l][n];
            const maxAct =
              d3.max(
                activations
                  .flat()
                  .filter(a => a !== undefined)
                  .map(Math.abs)
              ) || 0.1;
            const normAct = Math.abs(activation) / maxAct;

            return activation > 0
              ? `rgba(130, 195, 236, ${Math.min(0.95, 0.3 + normAct * 0.65)})`
              : `rgba(231, 137, 137, ${Math.min(0.95, 0.3 + normAct * 0.65)})`;
          }
        }
        return 'rgba(130, 195, 236, 0.3)';
      })
      .attr('stroke', function () {
        const classes = d3.select(this).attr('class');
        const match = classes.match(/layer-(\d+) neuron-(\d+)/);
        if (match) {
          const l = parseInt(match[1]);
          const n = parseInt(match[2]);

          if (activations[l] && activations[l][n] !== undefined) {
            const activation = activations[l][n];
            return activation > 0 ? '#9ecdf9' : '#f9c09e';
          }
        }
        return '#9ecdf9';
      })
      .attr('stroke-width', 2);

    // Reset all connections
    svg
      .selectAll('.connection')
      .attr('stroke-opacity', function () {
        return parseFloat(d3.select(this).attr('stroke-opacity')) * 0.7;
      })
      .attr('stroke-width', function () {
        return parseFloat(d3.select(this).attr('stroke-width')) * 0.8;
      });

    // Highlight selected neuron
    svg
      .select(`.neuron.layer-${layer}.neuron-${neuron}`)
      .attr('fill', 'rgba(34, 197, 94, 0.6)')
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 3)
      .attr('r', 24);

    // Highlight outgoing connections
    svg
      .selectAll(`.connection.layer-${layer}-${neuron}`)
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 3)
      .attr('stroke-opacity', 1);

    // Highlight incoming connections
    svg
      .selectAll('.connection')
      .filter(function () {
        const toLayer = d3.select(this).attr('data-to-layer');
        const toNeuron = d3.select(this).attr('data-to-neuron');
        return parseInt(toLayer) === layer && parseInt(toNeuron) === neuron;
      })
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 3)
      .attr('stroke-opacity', 1);

    // Log information to console
    if (activations[layer] && activations[layer][neuron] !== undefined) {
      setConsoleMessages(prev => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Selected neuron: Layer ${layer} Neuron ${neuron}, Activation = ${activations[
            layer
          ][neuron].toFixed(4)}`,
          type: 'info',
        },
      ]);
    }
  };

  // Split data into training and test sets
  const splitTrainTest = (data, test_ratio = 0.2) => {
    try {
      const numSamples = data.xs.shape[0];
      const numTest = Math.floor(numSamples * test_ratio);

      // Get data as arrays instead of tensors
      const xValues = data.xs.arraySync();
      const yValues = data.ys.arraySync();

      // Create shuffled indices without using tf.util functions
      const indices = Array.from({ length: numSamples }, (_, i) => i);

      // Simple Fisher-Yates shuffle
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Split the data manually using the shuffled indices
      const testXValues = [];
      const testYValues = [];
      const trainXValues = [];
      const trainYValues = [];

      for (let i = 0; i < numSamples; i++) {
        const idx = indices[i];
        if (i < numTest) {
          testXValues.push(xValues[idx]);
          testYValues.push(yValues[idx]);
        } else {
          trainXValues.push(xValues[idx]);
          trainYValues.push(yValues[idx]);
        }
      }

      // Convert back to tensors
      return {
        train: {
          xs: tf.tensor2d(trainXValues),
          ys: tf.tensor2d(trainYValues),
        },
        test: {
          xs: tf.tensor2d(testXValues),
          ys: tf.tensor2d(testYValues),
        },
      };
    } catch (error) {
      console.error('Error in splitTrainTest:', error);

      // Return a simple split as fallback (no shuffling)
      const xValues = data.xs.arraySync();
      const yValues = data.ys.arraySync();
      const numSamples = xValues.length;
      const splitIndex = Math.floor(numSamples * (1 - test_ratio));

      return {
        train: {
          xs: tf.tensor2d(xValues.slice(0, splitIndex)),
          ys: tf.tensor2d(yValues.slice(0, splitIndex)),
        },
        test: {
          xs: tf.tensor2d(xValues.slice(splitIndex)),
          ys: tf.tensor2d(yValues.slice(splitIndex)),
        },
      };
    }
  };

  // Evaluate model on test data
  const evaluateModel = async (model, testData) => {
    if (!model || !testData) return { loss: 0, accuracy: 0 };

    const result = await model.evaluate(testData.xs, testData.ys, {
      batchSize: Math.min(testData.xs.shape[0], 32),
      verbose: 0,
    });

    const loss = await result[0].data();
    const accuracy = await result[1].data();

    // Clean up tensors
    result[0].dispose();
    result[1].dispose();

    return { loss: loss[0], accuracy: accuracy[0] };
  };

  // Core simple training function that's guaranteed to work
  const simpleTraining = () => {
    console.log('Simple training function called!');

    if (isTraining) {
      console.log('Stopping training');
      setIsTraining(false);
      trainingControlRef.current = false;
      return;
    }

    try {
      // Start training
      console.log('Starting training');
      setIsTraining(true);
      trainingControlRef.current = true;

      // Clear previous data
      setLossHistory([]);
      setAccuracyHistory([]);
      setIterations(0);
      setLoss(0);

      // Generate dataset if needed
      if (!dataCache) {
        setConsoleMessages(prev => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            message: `Generating ${selectedDataset} dataset...`,
            type: 'info',
          },
        ]);

        generateDataset(selectedDataset)
          .then(data => {
            try {
              // Create model with the architecture
              const model = createModel();
              modelRef.current = model;

              // Initialize model layers for activation extraction
              initializeLayerOutputs(model);

              // Initialize weights and activations
              updateInitialWeightsAndActivations();
              visualizeNetwork();
              visualizeDataset();

              // Start training loop with initial delay
              setTimeout(runTrainingStep, 1000);
            } catch (error) {
              handleTrainingError('Error creating model', error);
            }
          })
          .catch(error => {
            handleTrainingError('Error generating dataset', error);
          });
      } else {
        try {
          // Create or reset model
          const model = createModel();
          modelRef.current = model;

          // Initialize model layers for activation extraction
          initializeLayerOutputs(model);

          // Update visualizations
          updateInitialWeightsAndActivations();
          visualizeNetwork();
          visualizeDataset();

          setTimeout(runTrainingStep, 1000);
        } catch (error) {
          handleTrainingError('Error setting up model', error);
        }
      }
    } catch (error) {
      handleTrainingError('Error starting training', error);
    }
  };

  // Initialize weights and activations with random values to avoid 0.000 display
  const updateInitialWeightsAndActivations = () => {
    // Initialize weights
    updateWeights();

    // Initialize activations with small random values
    const newActivations = [];
    for (let i = 0; i < layers.length; i++) {
      const layerActivations = [];
      for (let j = 0; j < layers[i]; j++) {
        // Small random values between -0.1 and 0.1
        layerActivations.push(Math.random() * 0.2 - 0.1);
      }
      newActivations.push(layerActivations);
    }
    setActivations(newActivations);

    // Log to console
    console.log('Initialized weights and activations');
  };

  // Helper function to set up layer outputs for activation extraction
  const initializeLayerOutputs = model => {
    if (!model) return;

    try {
      // Add a callback to all layers to track their activations
      for (let i = 0; i < model.layers.length; i++) {
        const layer = model.layers[i];
        // Save the original apply method
        const originalApply = layer.apply;

        // Override apply to capture outputs
        layer.apply = function (inputs, ...args) {
          const output = originalApply.call(this, inputs, ...args);
          // Store the output for later use
          layer.lastOutput = output;
          return output;
        };
      }
      console.log('Layer outputs tracking initialized');
    } catch (error) {
      console.error('Error initializing layer outputs:', error);
    }
  };

  // Extract activations from model using a single input sample
  const extractActivations = async (model, inputTensor) => {
    if (!model || !inputTensor) return null;

    try {
      // Get input layer activations (the input itself)
      const inputActivations = inputTensor.arraySync()[0];
      const allActivations = [inputActivations];

      // Forward pass through the network
      let currentInput = inputTensor;

      // For each layer, get its output
      for (let i = 0; i < model.layers.length; i++) {
        // Apply layer and get output
        const output = model.layers[i].apply(currentInput);
        const outputValues = output.arraySync()[0];

        // Add to activations list
        allActivations.push(outputValues);

        // Use this output as input to next layer
        currentInput = output;
      }

      // Clean up tensors
      if (currentInput !== inputTensor) {
        currentInput.dispose();
      }

      return allActivations;
    } catch (error) {
      console.error('Error extracting activations:', error);
      return null;
    }
  };

  // Handle training errors
  const handleTrainingError = (message, error) => {
    console.error(`${message}:`, error);
    setConsoleMessages(prev => [
      ...prev,
      {
        timestamp: new Date().toLocaleTimeString(),
        message: `${message}: ${error.message}`,
        type: 'error',
      },
    ]);
    setIsTraining(false);
    trainingControlRef.current = false;
  };

  // Training step function
  const runTrainingStep = async () => {
    if (!trainingControlRef.current) {
      console.log('Training stopped');
      return;
    }

    try {
      // Check if we have real data to train on
      if (!trainDataCache || !trainDataCache.xs) {
        // If no data yet, generate it first
        await generateDataset(selectedDataset);

        // If we still don't have data after generating, stop
        if (!trainDataCache || !trainDataCache.xs) {
          handleTrainingError(
            'Could not generate training data',
            new Error('Data generation failed')
          );
          return;
        }
      }

      // Create model if needed
      if (!modelRef.current) {
        modelRef.current = createModel();
        initializeLayerOutputs(modelRef.current);
        console.log('Created new model:', modelRef.current);
        setConsoleMessages(prev => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            message: `Created new neural network with architecture [${layers.join(', ')}]`,
            type: 'info',
          },
        ]);
      }

      // Perform one real training step with TensorFlow.js
      let history;
      try {
        // Actual training step
        history = await modelRef.current.fit(trainDataCache.xs, trainDataCache.ys, {
          epochs: 1,
          batchSize: Math.min(batchSize, trainDataCache.xs.shape[0]),
          verbose: 0,
        });

        // Get actual loss from model
        const currentLoss = history.history.loss[0];
        setLossHistory(prev => [...prev, currentLoss]);

        // Calculate accuracy if available
        if (history.history.acc) {
          setAccuracy(history.history.acc[0]);
          setAccuracyHistory(prev => [...prev, history.history.acc[0]]);
        }
      } catch (trainError) {
        console.error('Error during training step:', trainError);
        setConsoleMessages(prev => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            message: `Training error: ${trainError.message}`,
            type: 'error',
          },
        ]);
        // Continue with visualization updates even if training fails
      }

      // Get test loss by evaluating on test data
      if (testDataCache && testDataCache.xs) {
        try {
          const evalResult = await modelRef.current.evaluate(testDataCache.xs, testDataCache.ys, {
            verbose: 0,
          });

          if (evalResult && evalResult.length > 0) {
            const testLoss = evalResult[0].dataSync()[0];
            setLoss(testLoss);
          }
        } catch (evalError) {
          console.error('Error evaluating model:', evalError);
        }
      }

      // Increment iteration counter
      setIterations(prev => prev + 1);

      // Extract real weights from the model
      try {
        const realWeights = [];
        for (let i = 0; i < modelRef.current.layers.length; i++) {
          const layerWeights = modelRef.current.layers[i].getWeights();
          if (layerWeights && layerWeights.length > 0) {
            // Get the actual weight matrix (not biases)
            const weightMatrix = layerWeights[0].arraySync();
            realWeights.push(weightMatrix);
          }
        }
        setWeights(realWeights);
      } catch (weightError) {
        console.error('Error extracting weights:', weightError);
      }

      // Calculate real activations by running a single example through the network
      try {
        // Pick a random example from training data
        const idx = Math.floor(Math.random() * trainDataCache.xs.shape[0]);
        const singleInput = trainDataCache.xs.slice([idx], [1]);

        // Extract activations for all layers by running the example through the network
        const allActivations = await extractActivations(modelRef.current, singleInput);

        if (allActivations) {
          setActivations(allActivations);
        }

        // Clean up tensor
        singleInput.dispose();
      } catch (actError) {
        console.error('Error calculating activations:', actError);
      }

      // Update visualizations with the real data
      visualizeNetwork();
      updateChart(lossHistory, accuracyHistory);

      // Only update data visualization periodically to improve performance
      if (iterations % 5 === 0) {
        visualizeDataset();
      }

      // Add console message periodically to show progress
      if (iterations % 10 === 0) {
        const latestLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 0;
        setConsoleMessages(prev => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            message: `Iteration ${iterations}: Training Loss = ${latestLoss.toFixed(
              4
            )}, Test Loss = ${loss.toFixed(4)}`,
            type: 'info',
          },
        ]);
      }

      // Continue training if not reached max iterations
      if (iterations < 1000 && trainingControlRef.current) {
        const baseDelay = 500; // 500ms base delay (2 iterations per second at 1x speed)
        const delay = baseDelay / trainingSpeed;
        setTimeout(runTrainingStep, delay);
      } else {
        setIsTraining(false);
        trainingControlRef.current = false;
        setConsoleMessages(prev => [
          ...prev,
          {
            timestamp: new Date().toLocaleTimeString(),
            message: `Training completed after ${iterations} iterations. Final test loss: ${loss.toFixed(
              4
            )}`,
            type: 'success',
          },
        ]);
      }
    } catch (error) {
      handleTrainingError('Error in training step', error);
    }
  };

  // Function to visualize the dataset and decision boundary
  const visualizeDataset = async () => {
    if (!dataVisRef.current) return;

    const svg = d3.select(dataVisRef.current);
    svg.selectAll('*').remove();

    const width = dataVisRef.current.clientWidth || 400;
    const height = dataVisRef.current.clientHeight || 400;
    const padding = 40;

    // Set background
    svg.append('rect').attr('width', width).attr('height', height).attr('fill', '#0a0a0b');

    // Create scales with fixed extent to maintain consistency
    const xExtent = [-6, 6]; // Fixed range to match TensorFlow Playground
    const yExtent = [-6, 6];

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([padding, width - padding]);

    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([height - padding, padding]);

    // Draw grid
    const gridSize = 1;
    for (let x = Math.ceil(xExtent[0]); x <= Math.floor(xExtent[1]); x += gridSize) {
      svg
        .append('line')
        .attr('x1', xScale(x))
        .attr('y1', yScale(yExtent[0]))
        .attr('x2', xScale(x))
        .attr('y2', yScale(yExtent[1]))
        .attr('stroke', 'rgba(255, 255, 255, 0.05)')
        .attr('stroke-width', x === 0 ? 2 : 1);
    }

    for (let y = Math.ceil(yExtent[0]); y <= Math.floor(yExtent[1]); y += gridSize) {
      svg
        .append('line')
        .attr('x1', xScale(xExtent[0]))
        .attr('y1', yScale(y))
        .attr('x2', xScale(xExtent[1]))
        .attr('y2', yScale(y))
        .attr('stroke', 'rgba(255, 255, 255, 0.05)')
        .attr('stroke-width', y === 0 ? 2 : 1);
    }

    // Axis labels
    for (let x = Math.ceil(xExtent[0]); x <= Math.floor(xExtent[1]); x++) {
      if (x !== 0) {
        svg
          .append('text')
          .attr('x', xScale(x))
          .attr('y', yScale(0) + 15)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', 'rgba(255, 255, 255, 0.5)')
          .text(x);
      }
    }

    for (let y = Math.ceil(yExtent[0]); y <= Math.floor(yExtent[1]); y++) {
      if (y !== 0) {
        svg
          .append('text')
          .attr('x', xScale(0) - 15)
          .attr('y', yScale(y) + 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', 'rgba(255, 255, 255, 0.5)')
          .text(y);
      }
    }

    // Create grid of points for decision boundary
    const resolution = 40;
    const boundaryPoints = [];
    const stepX = (xExtent[1] - xExtent[0]) / resolution;
    const stepY = (yExtent[1] - yExtent[0]) / resolution;

    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const x = xExtent[0] + i * stepX;
        const y = yExtent[0] + j * stepY;
        boundaryPoints.push([x, y]);
      }
    }

    // Create predictions - always use the model if available
    let predictions = [];

    if (modelRef.current) {
      try {
        // Create tensor from boundary points
        const pointsTensor = tf.tensor2d(boundaryPoints);

        // Run forward pass through the model
        const predictionsTensor = modelRef.current.predict(pointsTensor);

        // Extract predictions as array
        predictions = Array.from(predictionsTensor.dataSync());

        // Clean up tensors
        pointsTensor.dispose();
        predictionsTensor.dispose();
      } catch (predError) {
        console.error('Error predicting decision boundary:', predError);
      }
    }

    // If we couldn't get predictions, generate default ones based on dataset type
    if (predictions.length === 0) {
      for (let i = 0; i < boundaryPoints.length; i++) {
        const [x, y] = boundaryPoints[i];

        // Create different default boundaries based on dataset type
        let value = 0.5;

        switch (selectedDataset) {
          case 'circle':
            // Circular boundary
            value = Math.sqrt(x * x + y * y) < 3 ? 0.8 : 0.2;
            break;
          case 'xor':
            // XOR boundary
            value = (x > 0 && y > 0) || (x < 0 && y < 0) ? 0.2 : 0.8;
            break;
          case 'spiral':
            // Simple approximation of spiral
            const angle = Math.atan2(y, x);
            const radius = Math.sqrt(x * x + y * y);
            value = (angle + radius) % (Math.PI * 2) < Math.PI ? 0.8 : 0.2;
            break;
          case 'gaussian':
            // Two gaussian clusters
            const dist1 = Math.sqrt((x + 2) * (x + 2) + (y + 2) * (y + 2));
            const dist2 = Math.sqrt((x - 2) * (x - 2) + (y - 2) * (y - 2));
            value = dist1 < dist2 ? 0.2 : 0.8;
            break;
          default:
            value = 0.5;
        }

        predictions.push(value);
      }
    }

    // Calculate the color scale
    const colorScale = d3
      .scaleLinear()
      .domain([0, 0.5, 1])
      .range(['#f97316', '#f0f0f0', '#3b82f6'])
      .interpolate(d3.interpolateRgb);

    // Draw the background gradient
    for (let i = 0; i < boundaryPoints.length; i++) {
      const [x, y] = boundaryPoints[i];
      const value = predictions[i];
      const finalValue = discretizeOutput ? (value > 0.5 ? 1 : 0) : value;

      svg
        .append('rect')
        .attr('x', xScale(x - stepX / 2))
        .attr('y', yScale(y + stepY / 2))
        .attr('width', xScale(x + stepX / 2) - xScale(x - stepX / 2))
        .attr('height', yScale(y - stepY / 2) - yScale(y + stepY / 2))
        .attr('fill', colorScale(finalValue))
        .attr('opacity', 0.5);
    }

    // Draw data points from dataset if available
    if (dataCache && dataCache.xs) {
      try {
        // Get data as arrays for visualization
        const dataArray = dataCache.xs.arraySync();
        const labelsArray = dataCache.ys.arraySync();

        for (let i = 0; i < dataArray.length; i++) {
          const x = dataArray[i][0];
          const y = dataArray[i][1];

          // Check if point is within our visible range
          if (x >= xExtent[0] && x <= xExtent[1] && y >= yExtent[0] && y <= yExtent[1]) {
            const label = labelsArray[i][0];

            svg
              .append('circle')
              .attr('cx', xScale(x))
              .attr('cy', yScale(y))
              .attr('r', 4)
              .attr('fill', label > 0.5 ? '#3b82f6' : '#f97316')
              .attr('stroke', '#ffffff')
              .attr('stroke-width', 1)
              .attr('opacity', 0.8);
          }
        }
      } catch (err) {
        console.error('Error showing dataset:', err);
      }
    }

    // Draw test data points if enabled
    if (showTestData && testDataCache && testDataCache.xs) {
      try {
        const testDataArray = testDataCache.xs.arraySync();
        const testLabelsArray = testDataCache.ys.arraySync();

        for (let i = 0; i < testDataArray.length; i++) {
          const x = testDataArray[i][0];
          const y = testDataArray[i][1];

          // Check if point is within our visible range
          if (x >= xExtent[0] && x <= xExtent[1] && y >= yExtent[0] && y <= yExtent[1]) {
            const label = testLabelsArray[i][0];

            svg
              .append('circle')
              .attr('cx', xScale(x))
              .attr('cy', yScale(y))
              .attr('r', 4)
              .attr('fill', 'none')
              .attr('stroke', label > 0.5 ? '#3b82f6' : '#f97316')
              .attr('stroke-width', 2)
              .attr('opacity', 0.8);
          }
        }
      } catch (err) {
        console.error('Error showing test data:', err);
      }
    }

    // Add color legend
    const legendWidth = 150;
    const legendHeight = 15;
    const legendX = width - legendWidth - 20;
    const legendY = height - 30;

    // Create gradient
    const defs = svg.append('defs');
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'colorGradient')
      .attr('x1', '0%')
      .attr('x2', '100%');

    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#f97316');

    gradient.append('stop').attr('offset', '50%').attr('stop-color', '#f0f0f0');

    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#3b82f6');

    // Draw legend rectangle
    svg
      .append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#colorGradient)');

    // Add legend labels
    svg
      .append('text')
      .attr('x', legendX)
      .attr('y', legendY - 5)
      .attr('text-anchor', 'start')
      .attr('font-size', '12px')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .text('Class 0');

    svg
      .append('text')
      .attr('x', legendX + legendWidth)
      .attr('y', legendY - 5)
      .attr('text-anchor', 'end')
      .attr('font-size', '12px')
      .attr('fill', 'rgba(255, 255, 255, 0.7)')
      .text('Class 1');
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <Brain size={24} color="#3b82f6" />
            <span style={styles.logoText}>Neural Network Playground</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button style={styles.iconButton} onClick={() => setShowHelp(true)}>
              <HelpCircle size={16} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div style={styles.mainContent}>
        {/* Sidebar */}
        <div style={styles.sidebar}>
          <div style={styles.tabs}>
            <div
              style={{ ...styles.tab, ...(activeTab === 'architecture' ? styles.activeTab : {}) }}
              onClick={() => setActiveTab('architecture')}
            >
              <Layers size={14} />
              <span>Architecture</span>
            </div>
            <div
              style={{ ...styles.tab, ...(activeTab === 'training' ? styles.activeTab : {}) }}
              onClick={() => setActiveTab('training')}
            >
              <Activity size={14} />
              <span>Training</span>
            </div>
            <div
              style={{ ...styles.tab, ...(activeTab === 'advanced' ? styles.activeTab : {}) }}
              onClick={() => setActiveTab('advanced')}
            >
              <Settings size={14} />
              <span>Advanced</span>
            </div>
          </div>

          <div style={styles.tabContent}>
            {activeTab === 'architecture' && (
              <div>
                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() => setExpandedSection(expandedSection === 'layers' ? '' : 'layers')}
                  >
                    <div style={styles.sectionTitle}>
                      <GitBranch
                        size={16}
                        color={expandedSection === 'layers' ? '#3b82f6' : '#9ca3af'}
                      />
                      <span>Network Layers</span>
                    </div>
                    {expandedSection === 'layers' ? (
                      <ChevronDown size={16} color="#3b82f6" />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                  </div>
                  {expandedSection === 'layers' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <Layers size={14} color="#9ca3af" />
                          <span>Layer Configuration</span>
                        </label>
                        <div style={styles.layerGrid}>
                          {layers.map((size, idx) => (
                            <React.Fragment key={idx}>
                              <div style={styles.layerInputWrapper}>
                                <div
                                  style={{
                                    fontSize: '11px',
                                    color: '#8b9cb7',
                                    marginBottom: '4px',
                                    textTransform: 'uppercase',
                                    fontWeight: '600',
                                    letterSpacing: '0.05em',
                                    textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
                                  }}
                                >
                                  {idx === 0
                                    ? 'Input'
                                    : idx === layers.length - 1
                                    ? 'Output'
                                    : `Hidden ${idx}`}
                                </div>
                                <input
                                  type="number"
                                  value={size}
                                  onChange={e => updateLayerSize(idx, e.target.value)}
                                  style={{
                                    ...styles.layerInput,
                                    borderColor:
                                      selectedNeuron?.layer === idx
                                        ? '#3b82f6'
                                        : 'rgba(255, 255, 255, 0.08)',
                                    background:
                                      selectedNeuron?.layer === idx
                                        ? 'rgba(59, 130, 246, 0.08)'
                                        : 'rgba(13, 13, 20, 0.6)',
                                    boxShadow:
                                      selectedNeuron?.layer === idx
                                        ? '0 0 10px rgba(59, 130, 246, 0.3)'
                                        : '0 2px 4px rgba(0, 0, 0, 0.2)',
                                  }}
                                  min="1"
                                  disabled={isTraining}
                                  onClick={() => setSelectedNeuron({ layer: idx, neuron: 0 })}
                                />
                              </div>
                              {idx < layers.length - 1 && <div style={styles.layerArrow}>→</div>}
                            </React.Fragment>
                          ))}
                        </div>
                      </div>
                      <div
                        style={{
                          ...styles.buttonGroup,
                          marginTop: '12px',
                        }}
                      >
                        <button
                          onClick={addLayer}
                          style={{
                            ...styles.button,
                            ...styles.primaryButton,
                            height: '40px',
                            borderRadius: '10px',
                            transition: 'all 0.2s',
                            position: 'relative',
                            overflow: 'hidden',
                          }}
                          disabled={isTraining}
                          onMouseEnter={e => {
                            e.currentTarget.style.transform = 'translateY(-2px)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.5)';
                          }}
                          onMouseLeave={e => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 2px 6px rgba(59, 130, 246, 0.4)';
                          }}
                        >
                          <Plus size={16} />
                          Add Layer
                        </button>
                        <button
                          onClick={removeLayer}
                          style={{
                            ...styles.button,
                            ...styles.secondaryButton,
                            height: '40px',
                            borderRadius: '10px',
                          }}
                          disabled={layers.length <= 2 || isTraining}
                          onMouseEnter={e => {
                            if (!e.currentTarget.disabled) {
                              e.currentTarget.style.background = 'rgba(30, 30, 40, 0.8)';
                              e.currentTarget.style.transform = 'translateY(-2px)';
                            }
                          }}
                          onMouseLeave={e => {
                            e.currentTarget.style.background = 'rgba(13, 13, 20, 0.6)';
                            e.currentTarget.style.transform = 'translateY(0)';
                          }}
                        >
                          <Minus size={16} />
                          Remove
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() =>
                      setExpandedSection(expandedSection === 'activation' ? '' : 'activation')
                    }
                  >
                    <div style={styles.sectionTitle}>
                      <Zap
                        size={16}
                        color={expandedSection === 'activation' ? '#3b82f6' : '#9ca3af'}
                      />
                      <span>Activation Function</span>
                    </div>
                    {expandedSection === 'activation' ? (
                      <ChevronDown size={16} color="#3b82f6" />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                  </div>
                  {expandedSection === 'activation' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <Activity size={14} color="#9ca3af" />
                          <span>Function Type</span>
                        </label>
                        <select
                          value={activationFunction}
                          onChange={e => setActivationFunction(e.target.value)}
                          style={{
                            ...styles.select,
                            height: '40px',
                            fontSize: '14px',
                            borderRadius: '8px',
                            background: 'rgba(13, 13, 20, 0.6)',
                            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                          }}
                          disabled={isTraining}
                        >
                          <option value="tanh">Tanh</option>
                          <option value="sigmoid">Sigmoid</option>
                          <option value="relu">ReLU</option>
                          <option value="linear">Linear</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>

                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() =>
                      setExpandedSection(expandedSection === 'dataset' ? '' : 'dataset')
                    }
                  >
                    <div style={styles.sectionTitle}>
                      <Database
                        size={16}
                        color={expandedSection === 'dataset' ? '#3b82f6' : '#9ca3af'}
                      />
                      <span>Dataset</span>
                    </div>
                    {expandedSection === 'dataset' ? (
                      <ChevronDown size={16} color="#3b82f6" />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                  </div>
                  {expandedSection === 'dataset' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <Database size={14} color="#9ca3af" />
                          <span>Dataset Type</span>
                        </label>
                        <select
                          value={selectedDataset}
                          onChange={e => setSelectedDataset(e.target.value)}
                          style={{
                            ...styles.select,
                            height: '40px',
                            fontSize: '14px',
                            borderRadius: '8px',
                            background: 'rgba(13, 13, 20, 0.6)',
                            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                          }}
                          disabled={isTraining}
                        >
                          <option value="circle">Circle</option>
                          <option value="xor">XOR</option>
                          <option value="gaussian">Gaussian</option>
                          <option value="spiral">Spiral</option>
                        </select>
                      </div>

                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <Sliders size={14} color="#9ca3af" />
                          <span>Problem Type</span>
                        </label>
                        <select
                          value={problemType}
                          onChange={e => setProblemType(e.target.value)}
                          style={{
                            ...styles.select,
                            height: '40px',
                            fontSize: '14px',
                            borderRadius: '8px',
                            background: 'rgba(13, 13, 20, 0.6)',
                            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                          }}
                          disabled={isTraining}
                        >
                          <option value="classification">Classification</option>
                          <option value="regression">Regression</option>
                        </select>
                      </div>

                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <TrendingUp size={14} color="#9ca3af" />
                          <span>
                            Noise Level: <span style={styles.sliderValue}>{noiseLevel}%</span>
                          </span>
                        </label>
                        <div style={{ position: 'relative', padding: '4px 0' }}>
                          <input
                            type="range"
                            min="0"
                            max="50"
                            step="1"
                            value={noiseLevel}
                            onChange={e => setNoiseLevel(parseInt(e.target.value))}
                            style={{
                              ...styles.slider,
                              background: `linear-gradient(to right, #60a5fa ${
                                noiseLevel * 2
                              }%, rgba(13, 13, 20, 0.6) ${noiseLevel * 2}%)`,
                            }}
                            disabled={isTraining}
                          />
                        </div>
                      </div>

                      <div style={styles.formGroup}>
                        <label style={styles.label}>
                          <Shuffle size={14} color="#9ca3af" />
                          <span>
                            Test Ratio:{' '}
                            <span style={styles.sliderValue}>{Math.round(testRatio * 100)}%</span>
                          </span>
                        </label>
                        <div style={{ position: 'relative', padding: '4px 0' }}>
                          <input
                            type="range"
                            min="0.1"
                            max="0.5"
                            step="0.05"
                            value={testRatio}
                            onChange={e => setTestRatio(parseFloat(e.target.value))}
                            style={{
                              ...styles.slider,
                              background: `linear-gradient(to right, #60a5fa ${
                                testRatio * 200
                              }%, rgba(13, 13, 20, 0.6) ${testRatio * 200}%)`,
                            }}
                            disabled={isTraining}
                          />
                        </div>
                      </div>

                      <div style={styles.formGroup}>
                        <label
                          style={{
                            ...styles.label,
                            display: 'flex',
                            alignItems: 'center',
                            padding: '6px 10px',
                            background: 'rgba(13, 13, 20, 0.4)',
                            borderRadius: '8px',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            cursor: 'pointer',
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={showTestData}
                            onChange={e => setShowTestData(e.target.checked)}
                            disabled={isTraining}
                            style={{ marginRight: '8px' }}
                          />
                          <Eye size={14} color="#9ca3af" style={{ marginRight: '6px' }} />
                          <span>Show test data</span>
                        </label>
                      </div>

                      <div style={styles.formGroup}>
                        <label
                          style={{
                            ...styles.label,
                            display: 'flex',
                            alignItems: 'center',
                            padding: '6px 10px',
                            background: 'rgba(13, 13, 20, 0.4)',
                            borderRadius: '8px',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            cursor: 'pointer',
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={discretizeOutput}
                            onChange={e => setDiscretizeOutput(e.target.checked)}
                            disabled={isTraining}
                            style={{ marginRight: '8px' }}
                          />
                          <Box size={14} color="#9ca3af" style={{ marginRight: '6px' }} />
                          <span>Discretize output</span>
                        </label>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'training' && (
              <div>
                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() =>
                      setExpandedSection(expandedSection === 'optimizer' ? '' : 'optimizer')
                    }
                  >
                    <div style={styles.sectionTitle}>
                      <TrendingUp size={14} />
                      <span>Optimizer</span>
                    </div>
                    {expandedSection === 'optimizer' ? (
                      <ChevronDown size={14} />
                    ) : (
                      <ChevronRight size={14} />
                    )}
                  </div>
                  {expandedSection === 'optimizer' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>Algorithm</label>
                        <select
                          value={optimizer}
                          onChange={e => setOptimizer(e.target.value)}
                          style={styles.select}
                          disabled={isTraining}
                        >
                          <option value="sgd">SGD</option>
                          <option value="adam">Adam</option>
                          <option value="rmsprop">RMSprop</option>
                        </select>
                      </div>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>Learning Rate</label>
                        <select
                          value={learningRate}
                          onChange={e => setLearningRate(parseFloat(e.target.value))}
                          style={styles.select}
                          disabled={isTraining}
                        >
                          <option value="0.001">0.001</option>
                          <option value="0.003">0.003</option>
                          <option value="0.01">0.01</option>
                          <option value="0.03">0.03</option>
                          <option value="0.1">0.1</option>
                          <option value="0.3">0.3</option>
                          <option value="1">1.0</option>
                        </select>
                      </div>
                      {optimizer === 'sgd' && (
                        <div style={styles.formGroup}>
                          <label style={styles.label}>Momentum: {momentum.toFixed(2)}</label>
                          <input
                            type="range"
                            min="0"
                            max="0.99"
                            step="0.01"
                            value={momentum}
                            onChange={e => setMomentum(parseFloat(e.target.value))}
                            style={styles.slider}
                            disabled={isTraining}
                          />
                        </div>
                      )}
                      <div style={styles.formGroup}>
                        <label style={styles.label}>Batch Size: {batchSize}</label>
                        <input
                          type="range"
                          min="1"
                          max="30"
                          step="1"
                          value={batchSize}
                          onChange={e => setBatchSize(parseInt(e.target.value))}
                          style={styles.slider}
                          disabled={isTraining}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'advanced' && (
              <div>
                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() =>
                      setExpandedSection(
                        expandedSection === 'regularization' ? '' : 'regularization'
                      )
                    }
                  >
                    <div style={styles.sectionTitle}>
                      <Sliders size={14} />
                      <span>Regularization</span>
                    </div>
                    {expandedSection === 'regularization' ? (
                      <ChevronDown size={14} />
                    ) : (
                      <ChevronRight size={14} />
                    )}
                  </div>
                  {expandedSection === 'regularization' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <label style={styles.label}>Type</label>
                        <select
                          value={regularizationType}
                          onChange={e => setRegularizationType(e.target.value)}
                          style={styles.select}
                          disabled={isTraining}
                        >
                          <option value="none">None</option>
                          <option value="l1">L1</option>
                          <option value="l2">L2</option>
                          <option value="l1l2">L1 + L2</option>
                        </select>
                      </div>
                      {regularizationType !== 'none' && (
                        <div style={styles.formGroup}>
                          <label style={styles.label}>Rate</label>
                          <select
                            value={regularizationRate}
                            onChange={e => setRegularizationRate(parseFloat(e.target.value))}
                            style={styles.select}
                            disabled={isTraining}
                          >
                            <option value="0">0</option>
                            <option value="0.001">0.001</option>
                            <option value="0.003">0.003</option>
                            <option value="0.01">0.01</option>
                            <option value="0.03">0.03</option>
                            <option value="0.1">0.1</option>
                          </select>
                        </div>
                      )}
                      <div style={styles.formGroup}>
                        <label style={styles.label}>Dropout Rate: {dropoutRate.toFixed(2)}</label>
                        <input
                          type="range"
                          min="0"
                          max="0.5"
                          step="0.05"
                          value={dropoutRate}
                          onChange={e => setDropoutRate(parseFloat(e.target.value))}
                          style={styles.slider}
                          disabled={isTraining}
                        />
                      </div>
                    </div>
                  )}
                </div>

                <div style={styles.section}>
                  <div
                    style={styles.sectionHeader}
                    onClick={() =>
                      setExpandedSection(
                        expandedSection === 'initialization' ? '' : 'initialization'
                      )
                    }
                  >
                    <div style={styles.sectionTitle}>
                      <Box size={14} />
                      <span>Weight Initialization</span>
                    </div>
                    {expandedSection === 'initialization' ? (
                      <ChevronDown size={14} />
                    ) : (
                      <ChevronRight size={14} />
                    )}
                  </div>
                  {expandedSection === 'initialization' && (
                    <div style={styles.sectionContent}>
                      <div style={styles.formGroup}>
                        <select
                          value={weightInitialization}
                          onChange={e => setWeightInitialization(e.target.value)}
                          style={styles.select}
                          disabled={isTraining}
                        >
                          <option value="xavier">Xavier</option>
                          <option value="he">He</option>
                          <option value="random">Random</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Training Controls */}
          <div style={styles.trainingControls}>
            <div
              style={{
                marginBottom: '10px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div style={{ fontSize: '14px', color: '#e5e7eb' }}>
                Epoch: {iterations.toString().padStart(3, '0')}
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  style={{ ...styles.iconButton, color: '#22c55e' }}
                  onClick={() => {
                    setDataCache(null); // Force dataset regeneration
                    setLossHistory([]);
                    setAccuracyHistory([]);
                    setIterations(0);
                  }}
                >
                  <Shuffle size={14} />
                </button>
              </div>
            </div>

            <button
              onClick={() => {
                console.log('Training button clicked');
                simpleTraining();
              }}
              style={{
                ...styles.trainButton,
                background: isTraining ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                boxShadow: '0 0 10px rgba(0, 0, 0, 0.3)',
                ...(isTraining
                  ? {
                      borderColor: '#ef4444',
                      color: '#ef4444',
                    }
                  : {
                      borderColor: '#22c55e',
                      color: '#22c55e',
                    }),
              }}
              onMouseEnter={e => {
                if (isTraining) {
                  e.target.style.background = 'rgba(239, 68, 68, 0.3)';
                  e.target.style.borderColor = '#ef4444';
                  e.target.style.color = '#ef4444';
                } else {
                  e.target.style.background = 'rgba(34, 197, 94, 0.3)';
                  e.target.style.borderColor = '#22c55e';
                  e.target.style.color = '#22c55e';
                }
              }}
              onMouseLeave={e => {
                if (isTraining) {
                  e.target.style.background = 'rgba(239, 68, 68, 0.2)';
                  e.target.style.borderColor = '#ef4444';
                  e.target.style.color = '#ef4444';
                } else {
                  e.target.style.background = 'rgba(34, 197, 94, 0.2)';
                  e.target.style.borderColor = '#22c55e';
                  e.target.style.color = '#22c55e';
                }
              }}
            >
              {isTraining ? (
                <>
                  <Pause size={16} />
                  <span>Stop Training</span>
                </>
              ) : (
                <>
                  <Play size={16} />
                  <span>Start Training</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Visualization Area */}
        <div style={styles.visualizationArea}>
          {/* Metrics Bar */}
          <div style={styles.metricsBar}>
            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Test Loss</div>
              <div style={styles.metricValue}>{loss.toFixed(4)}</div>
              <div style={{ ...styles.metricChange, ...styles.metricNegative }}>
                <TrendingUp size={12} />
                {lossHistory.length > 1
                  ? (
                      (lossHistory[lossHistory.length - 1] - lossHistory[lossHistory.length - 2]) *
                      100
                    ).toFixed(2)
                  : '0.00'}
                %
              </div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Training Loss</div>
              <div style={styles.metricValue}>
                {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '0.000'}
              </div>
              <div style={{ ...styles.metricChange, ...styles.metricNegative }}>
                <TrendingUp size={12} />
                {lossHistory.length > 1
                  ? 'Δ' +
                    (
                      lossHistory[lossHistory.length - 1] - lossHistory[lossHistory.length - 2]
                    ).toFixed(4)
                  : 'N/A'}
              </div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Iterations</div>
              <div style={styles.metricValue}>{iterations}</div>
              <div style={styles.metricChange}>
                {isTraining && (
                  <Loader size={12} style={{ animation: 'spin 2s linear infinite' }} />
                )}
                {isTraining ? 'Training...' : 'Ready'}
              </div>
            </div>

            <div style={styles.metricCard}>
              <div style={styles.metricLabel}>Speed</div>
              <div style={styles.metricValue}>{trainingSpeed}x</div>
              <div style={styles.metricChange}>
                <input
                  type="range"
                  min="0.1"
                  max="5"
                  step="0.1"
                  value={trainingSpeed}
                  onChange={e => setTrainingSpeed(parseFloat(e.target.value))}
                  style={{ ...styles.slider, width: '100%', marginTop: '4px' }}
                />
              </div>
            </div>
          </div>

          {/* Visualization Panels */}
          <div style={styles.visualizationPanels}>
            {/* Network Visualization */}
            <div style={styles.networkPanel}>
              <div style={styles.panelHeader}>
                <div style={styles.panelTitle}>
                  <Network size={14} />
                  <span>Network Architecture</span>
                </div>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {selectedNeuron && (
                    <span style={{ fontSize: '12px', color: '#22c55e' }}>
                      Layer {selectedNeuron.layer}, Neuron {selectedNeuron.neuron}
                    </span>
                  )}
                </div>
              </div>
              <div style={styles.networkContainer}>
                <svg ref={svgRef} style={styles.networkSvg}></svg>
              </div>
            </div>

            {/* Data Visualization */}
            <div style={styles.dataPanel}>
              <div style={styles.panelHeader}>
                <div style={styles.panelTitle}>
                  <Activity size={14} />
                  <span>OUTPUT</span>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <div style={styles.checkboxGroup}>
                    <input
                      type="checkbox"
                      id="showTestData"
                      checked={showTestData}
                      onChange={e => setShowTestData(e.target.checked)}
                      style={styles.checkbox}
                    />
                    <label htmlFor="showTestData" style={styles.checkboxLabel}>
                      Show test data
                    </label>
                  </div>
                  <div style={styles.checkboxGroup}>
                    <input
                      type="checkbox"
                      id="discretizeOutput"
                      checked={discretizeOutput}
                      onChange={e => setDiscretizeOutput(e.target.checked)}
                      style={styles.checkbox}
                    />
                    <label htmlFor="discretizeOutput" style={styles.checkboxLabel}>
                      Discretize output
                    </label>
                  </div>
                </div>
              </div>
              <div style={styles.dataContainer}>
                <svg ref={dataVisRef} style={styles.dataSvg}></svg>
              </div>
            </div>
          </div>

          {/* Chart Panel */}
          <div style={styles.chartPanel}>
            <div style={styles.panelHeader}>
              <div style={styles.panelTitle}>
                <BarChart3 size={14} />
                <span>Training Progress</span>
              </div>
            </div>
            <div style={styles.chartContainer}>
              <svg ref={chartRef} style={styles.chart}></svg>
            </div>
          </div>

          {/* Console */}
          <div style={styles.console}>
            <div style={styles.consoleHeader}>
              <span>Console Output</span>
              <button style={styles.iconButton} onClick={() => setConsoleMessages([])}>
                Clear
              </button>
            </div>
            <div style={styles.consoleContent}>
              {consoleMessages.map((msg, idx) => (
                <div key={idx} style={styles.consoleLine}>
                  <span style={styles.consoleTimestamp}>[{msg.timestamp}]</span>{' '}
                  <span
                    style={{
                      color:
                        msg.type === 'error'
                          ? '#ef4444'
                          : msg.type === 'success'
                          ? '#22c55e'
                          : msg.type === 'warning'
                          ? '#f59e0b'
                          : '#e5e7eb',
                    }}
                  >
                    {msg.message}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Help Modal */}
      <div style={styles.helpOverlay} onClick={() => setShowHelp(false)}>
        <div style={styles.helpPanel} onClick={e => e.stopPropagation()}>
          <div style={styles.helpHeader}>
            <div style={styles.helpTitle}>
              <HelpCircle size={20} />
              <span>Neural Network Visualizer</span>
            </div>
            <button style={styles.iconButton} onClick={() => setShowHelp(false)}>
              <X size={16} />
            </button>
          </div>
          <div style={styles.helpContent}>
            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>About This Application</div>
              <div style={styles.helpText}>
                This application provides an interactive, production-grade neural network training
                environment powered by TensorFlow.js. It allows you to configure, train, and
                visualize deep neural networks in real-time directly in your browser, offering a
                transparent view into the internal mechanics of how neural networks learn.
              </div>
            </div>

            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>Technical Implementation</div>
              <div style={styles.helpText}>
                The entire training process is handled by TensorFlow.js with no simplified
                abstractions:
                <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                  <li>Full gradient descent optimization with backpropagation</li>
                  <li>Direct extraction of weights and activations from model layers</li>
                  <li>Real-time visualization of actual neuron states and connection weights</li>
                  <li>Proper training/test data splitting with performance evaluation</li>
                  <li>
                    Support for various activation functions, optimizers, and regularization
                    techniques
                  </li>
                  <li>Implementation of industry-standard neural network training practices</li>
                </ul>
              </div>
            </div>

            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>Network Visualization Technology</div>
              <div style={styles.helpText}>
                The visualization displays the actual state of neurons and connections during
                training:
                <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                  <li>Neuron colors and intensities reflect actual activation values</li>
                  <li>Connection thicknesses correspond to weight magnitudes</li>
                  <li>
                    Blue connections represent positive weights, red connections represent negative
                    weights
                  </li>
                  <li>
                    All visualized metrics are directly extracted from the TensorFlow.js model
                  </li>
                  <li>
                    Visualization updates in real-time during training to show the learning process
                  </li>
                </ul>
              </div>
            </div>

            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>Educational Value</div>
              <div style={styles.helpText}>
                This tool is designed for both educational and practical purposes:
                <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                  <li>Understand the dynamics of neural network training</li>
                  <li>Experiment with hyperparameters and observe their effects</li>
                  <li>Visualize how networks solve different types of problems</li>
                  <li>Explore concepts like overfitting, regularization, and generalization</li>
                  <li>Gain intuition about neural network behavior through visual feedback</li>
                </ul>
              </div>
            </div>

            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>Implemented Datasets</div>
              <div style={styles.helpText}>
                <p style={{ marginBottom: '8px' }}>
                  The application provides several classic machine learning problems:
                </p>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>XOR:</td>
                      <td style={{ padding: '8px' }}>
                        The classical logical XOR problem that requires a hidden layer to solve.
                        Demonstrates the importance of non-linear transformations in neural
                        networks.
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>Circle:</td>
                      <td style={{ padding: '8px' }}>
                        Binary classification of points inside or outside a circle. Demonstrates the
                        network's ability to learn circular decision boundaries.
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>Gaussian:</td>
                      <td style={{ padding: '8px' }}>
                        Classification of points from two Gaussian distributions. Shows how networks
                        handle overlapping distributions and probability densities.
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>Spiral:</td>
                      <td style={{ padding: '8px' }}>
                        Classification of points from intertwined spiral patterns. Demonstrates the
                        power of neural networks to learn complex, non-linear decision boundaries.
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div style={styles.helpSection}>
              <div style={styles.helpSectionTitle}>Advanced Configuration Options</div>
              <div style={styles.helpText}>
                The system allows detailed configuration of network architecture and training:
                <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                  <li>
                    <strong>Architecture:</strong> Add, remove, and resize layers to create custom
                    network topologies
                  </li>
                  <li>
                    <strong>Activation Functions:</strong> Choose from ReLU, Sigmoid, Tanh, and
                    Linear activations
                  </li>
                  <li>
                    <strong>Optimizers:</strong> SGD with momentum, Adam, and RMSProp with
                    configurable learning rates
                  </li>
                  <li>
                    <strong>Regularization:</strong> L1, L2, and Dropout options to prevent
                    overfitting
                  </li>
                  <li>
                    <strong>Weight Initialization:</strong> Xavier, He, and Random initialization
                    strategies
                  </li>
                  <li>
                    <strong>Training Parameters:</strong> Batch size, noise level, and test/train
                    ratio
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;
