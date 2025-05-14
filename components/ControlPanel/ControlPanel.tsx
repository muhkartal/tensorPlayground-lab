import React, { useState } from 'react';
import './ControlPanel.css';

interface ControlPanelProps {
  onStartTraining: () => void;
  onStopTraining: () => void;
  onReset: () => void;
  onParameterChange: (param: string, value: number | string) => void;
  isTraining: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  onStartTraining,
  onStopTraining,
  onReset,
  onParameterChange,
  isTraining,
}) => {
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [epochs, setEpochs] = useState<number>(100);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [optimizer, setOptimizer] = useState<string>('adam');
  const [regularization, setRegularization] = useState<string>('none');
  const [regStrength, setRegStrength] = useState<number>(0.001);

  const handleLearningRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setLearningRate(value);
    onParameterChange('learningRate', value);
  };

  const handleEpochsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    setEpochs(value);
    onParameterChange('epochs', value);
  };
  
  const handleBatchSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    setBatchSize(value);
    onParameterChange('batchSize', value);
  };
  
  const handleOptimizerChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setOptimizer(value);
    onParameterChange('optimizer', value);
  };
  
  const handleRegularizationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setRegularization(value);
    onParameterChange('regularization', value);
  };
  
  const handleRegStrengthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setRegStrength(value);
    onParameterChange('regStrength', value);
  };

  return (
    <div className="control-panel">
      <h2>Training Parameters</h2>
      
      <div className="parameter-group">
        <label htmlFor="learning-rate">Learning Rate:</label>
        <input 
          id="learning-rate"
          type="range" 
          min="0.0001" 
          max="0.1" 
          step="0.0001" 
          value={learningRate} 
          onChange={handleLearningRateChange} 
        />
        <span className="parameter-value">{learningRate}</span>
      </div>
      
      <div className="parameter-group">
        <label htmlFor="epochs">Epochs:</label>
        <input 
          id="epochs"
          type="number" 
          min="1" 
          max="1000" 
          value={epochs} 
          onChange={handleEpochsChange} 
        />
      </div>
      
      <div className="parameter-group">
        <label htmlFor="batch-size">Batch Size:</label>
        <input 
          id="batch-size"
          type="number" 
          min="1" 
          max="128" 
          value={batchSize} 
          onChange={handleBatchSizeChange} 
        />
      </div>
      
      <div className="parameter-group">
        <label htmlFor="optimizer">Optimizer:</label>
        <select 
          id="optimizer"
          value={optimizer} 
          onChange={handleOptimizerChange}
        >
          <option value="sgd">SGD</option>
          <option value="adam">Adam</option>
          <option value="rmsprop">RMSprop</option>
          <option value="adagrad">AdaGrad</option>
        </select>
      </div>
      
      <div className="parameter-group">
        <label htmlFor="regularization">Regularization:</label>
        <select 
          id="regularization"
          value={regularization} 
          onChange={handleRegularizationChange}
        >
          <option value="none">None</option>
          <option value="l1">L1</option>
          <option value="l2">L2</option>
          <option value="elasticnet">Elastic Net</option>
        </select>
      </div>
      
      {regularization !== 'none' && (
        <div className="parameter-group">
          <label htmlFor="reg-strength">Regularization Strength:</label>
          <input 
            id="reg-strength"
            type="range" 
            min="0.0001" 
            max="0.01" 
            step="0.0001" 
            value={regStrength} 
            onChange={handleRegStrengthChange} 
          />
          <span className="parameter-value">{regStrength}</span>
        </div>
      )}
      
      <div className="button-group">
        {!isTraining ? (
          <button className="start-button" onClick={onStartTraining}>
            Start Training
          </button>
        ) : (
          <button className="stop-button" onClick={onStopTraining}>
            Stop Training
          </button>
        )}
        <button className="reset-button" onClick={onReset}>
          Reset
        </button>
      </div>
    </div>
  );
};

export default ControlPanel; 