import { useCallback, useState } from 'react';

interface NetworkConfig {
  layers: number[];
  activations: string[];
}

export const useWebGLRenderer = () => {
  const [gl, setGl] = useState<WebGLRenderingContext | null>(null);
  const [program, setProgram] = useState<WebGLProgram | null>(null);

  // Initialize WebGL context and shaders
  const initializeRenderer = useCallback((canvas: HTMLCanvasElement) => {
    // Get WebGL context
    const glContext = canvas.getContext('webgl2') as WebGLRenderingContext;
    
    if (!glContext) {
      console.error('WebGL 2 not supported');
      return;
    }
    
    // Create shaders and program
    const vertexShaderSource = `
      attribute vec2 aPosition;
      attribute vec3 aColor;
      varying vec3 vColor;
      
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
        vColor = aColor;
      }
    `;
    
    const fragmentShaderSource = `
      precision mediump float;
      varying vec3 vColor;
      
      void main() {
        gl_FragColor = vec4(vColor, 1.0);
      }
    `;
    
    // Create and compile vertex shader
    const vertexShader = glContext.createShader(glContext.VERTEX_SHADER);
    if (!vertexShader) {
      console.error('Failed to create vertex shader');
      return;
    }
    
    glContext.shaderSource(vertexShader, vertexShaderSource);
    glContext.compileShader(vertexShader);
    
    // Create and compile fragment shader
    const fragmentShader = glContext.createShader(glContext.FRAGMENT_SHADER);
    if (!fragmentShader) {
      console.error('Failed to create fragment shader');
      return;
    }
    
    glContext.shaderSource(fragmentShader, fragmentShaderSource);
    glContext.compileShader(fragmentShader);
    
    // Create program and link shaders
    const shaderProgram = glContext.createProgram();
    if (!shaderProgram) {
      console.error('Failed to create shader program');
      return;
    }
    
    glContext.attachShader(shaderProgram, vertexShader);
    glContext.attachShader(shaderProgram, fragmentShader);
    glContext.linkProgram(shaderProgram);
    
    // Check for shader compilation and linking errors
    if (!glContext.getShaderParameter(vertexShader, glContext.COMPILE_STATUS)) {
      console.error('Vertex shader compilation failed:', 
                  glContext.getShaderInfoLog(vertexShader));
      return;
    }
    
    if (!glContext.getShaderParameter(fragmentShader, glContext.COMPILE_STATUS)) {
      console.error('Fragment shader compilation failed:', 
                  glContext.getShaderInfoLog(fragmentShader));
      return;
    }
    
    if (!glContext.getProgramParameter(shaderProgram, glContext.LINK_STATUS)) {
      console.error('Shader program linking failed:', 
                  glContext.getProgramInfoLog(shaderProgram));
      return;
    }
    
    glContext.useProgram(shaderProgram);
    
    // Set clear color and enable depth testing
    glContext.clearColor(0.95, 0.95, 0.95, 1.0);
    glContext.enable(glContext.DEPTH_TEST);
    
    setGl(glContext);
    setProgram(shaderProgram);
  }, []);
  
  // Render the neural network with WebGL
  const renderNetwork = useCallback((
    networkConfig: NetworkConfig,
    weights: number[][][],
    activationValues: number[][],
    gradients?: number[][][]
  ) => {
    if (!gl || !program) return;
    
    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    // Implement actual network rendering here
    // This is a simplified placeholder
    console.log('Rendering network with:', 
      networkConfig, 
      `Weights shape: ${weights.length} layers`,
      `Activations shape: ${activationValues.length} layers`
    );
    
    // In a real implementation, we would:
    // 1. Create geometry for neurons (circles/spheres)
    // 2. Create geometry for connections (lines)
    // 3. Color based on activation values and weights
    // 4. Render with proper transformations
    
  }, [gl, program]);
  
  return {
    initializeRenderer,
    renderNetwork,
  };
}; 