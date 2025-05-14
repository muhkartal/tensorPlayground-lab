import React, { useState } from 'react';
import NetworkVisualizer from './components/NetworkVisualizer';
import ControlPanel from './components/ControlPanel';
import './App.css';

/**
 * Main application component
 * 
 * This is the root component of the application that renders
 * the Neural Network Visualizer.
 * 
 * @returns {React.ReactElement} The rendered application
 */
function App() {
  const [networkConfig, setNetworkConfig] = useState({
    layers: [2, 5, 5, 2],
    activations: ['relu', 'relu', 'sigmoid']
  });
  
  const [isTraining, setIsTraining] = useState(false);
  
  const handleStartTraining = () => {
    setIsTraining(true);
  };
  
  const handleStopTraining = () => {
    setIsTraining(false);
  };
  
  const handleReset = () => {
    setIsTraining(false);
    // Reset network state
  };
  
  const handleParameterChange = (param, value) => {
    console.log(`Parameter ${param} changed to ${value}`);
    // Update parameters
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Network Playground</h1>
      </header>
      <main className="App-main">
        <div className="visualizer-container">
          <NetworkVisualizer networkConfig={networkConfig} />
        </div>
        <div className="control-panel-container">
          <ControlPanel 
            onStartTraining={handleStartTraining}
            onStopTraining={handleStopTraining}
            onReset={handleReset}
            onParameterChange={handleParameterChange}
            isTraining={isTraining}
          />
        </div>
      </main>
    </div>
  );
}

export default App;