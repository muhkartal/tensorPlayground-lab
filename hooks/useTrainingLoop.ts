import { useCallback, useRef, useState } from 'react';

interface TrainingLoopOptions {
  onStep?: (step: number) => void;
  onComplete?: () => void;
  fps?: number;
}

export const useTrainingLoop = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const animationRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const stepsRef = useRef<number>(0);
  const totalStepsRef = useRef<number>(0);
  const callbackRef = useRef<TrainingLoopOptions>({});
  
  // Start the animation loop
  const startTrainingLoop = useCallback((totalSteps: number, options?: TrainingLoopOptions) => {
    if (isRunning) return;
    
    setIsRunning(true);
    setCurrentStep(0);
    stepsRef.current = 0;
    totalStepsRef.current = totalSteps;
    callbackRef.current = options || {};
    lastTimeRef.current = 0;
    
    // Cancel any existing animation frame
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    // Start the animation loop
    const fps = options?.fps || 60;
    const frameInterval = 1000 / fps;
    
    const animate = (timestamp: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      
      const elapsed = timestamp - lastTimeRef.current;
      
      if (elapsed > frameInterval) {
        // Update step
        const step = stepsRef.current;
        setCurrentStep(step);
        
        // Call onStep callback
        if (callbackRef.current.onStep) {
          callbackRef.current.onStep(step);
        }
        
        // Increment step counter
        stepsRef.current++;
        lastTimeRef.current = timestamp;
        
        // Check if training is complete
        if (stepsRef.current >= totalStepsRef.current) {
          stopTrainingLoop();
          return;
        }
      }
      
      // Continue animation loop
      animationRef.current = requestAnimationFrame(animate);
    };
    
    // Start animation
    animationRef.current = requestAnimationFrame(animate);
    
  }, [isRunning]);
  
  // Stop the animation loop
  const stopTrainingLoop = useCallback(() => {
    if (!isRunning) return;
    
    // Cancel animation frame
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = 0;
    }
    
    setIsRunning(false);
    
    // Call onComplete callback
    if (callbackRef.current.onComplete) {
      callbackRef.current.onComplete();
    }
    
  }, [isRunning]);
  
  // Pause the animation loop
  const pauseTrainingLoop = useCallback(() => {
    if (!isRunning) return;
    
    // Cancel animation frame
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = 0;
    }
    
    setIsRunning(false);
  }, [isRunning]);
  
  // Resume the animation loop
  const resumeTrainingLoop = useCallback(() => {
    if (isRunning || stepsRef.current >= totalStepsRef.current) return;
    
    setIsRunning(true);
    lastTimeRef.current = 0;
    
    // Continue with the animation loop
    const fps = callbackRef.current?.fps || 60;
    const frameInterval = 1000 / fps;
    
    const animate = (timestamp: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      
      const elapsed = timestamp - lastTimeRef.current;
      
      if (elapsed > frameInterval) {
        // Update step
        const step = stepsRef.current;
        setCurrentStep(step);
        
        // Call onStep callback
        if (callbackRef.current.onStep) {
          callbackRef.current.onStep(step);
        }
        
        // Increment step counter
        stepsRef.current++;
        lastTimeRef.current = timestamp;
        
        // Check if training is complete
        if (stepsRef.current >= totalStepsRef.current) {
          stopTrainingLoop();
          return;
        }
      }
      
      // Continue animation loop
      animationRef.current = requestAnimationFrame(animate);
    };
    
    // Start animation
    animationRef.current = requestAnimationFrame(animate);
    
  }, [isRunning, stopTrainingLoop]);
  
  // Clean up on unmount
  const cleanup = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = 0;
    }
  }, []);
  
  return {
    startTrainingLoop,
    stopTrainingLoop,
    pauseTrainingLoop,
    resumeTrainingLoop,
    cleanup,
    isRunning,
    currentStep,
    progress: totalStepsRef.current > 0 
      ? stepsRef.current / totalStepsRef.current 
      : 0,
  };
}; 