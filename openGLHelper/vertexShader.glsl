#version 150

// input: vertices in object coordinates and per-vetex attributes
in vec3 position;  // in is a parameter qualifier: which means copy in, but don't copy out (read-only)
in vec4 color; 
in vec4 colorFromSecondImage;
// additional input used in shader applied to the smooth mode 
in vec3 positionCenter;  // inout is another parameter qualifier, which means copy in and copy out
in vec3 positionLeft;
in vec3 positionRight;
in vec3 positionDown;
in vec3 positionUp;

// output: gl_Position is a built-in output variable (as suggested by gl_ prefix) and thus no need to specify. 
// out is a parameter qualifier, which means only copy out (to other shaders/pixels)
out vec4 col; 

// constant info passed to shader, and therefor read-only in the shader:
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float scale, exponent; 
uniform int controlRenderingMode;
// for extra credits 1:
uniform int lightingMode;
uniform int jetColorMapMode;
uniform int numBytesPerPixel;
uniform int secondImageMode; 
uniform int secondImageValid; 
uniform int layerMode; 



// global variables (intermediary vectors) 
vec3 smoothPosition;  // average positios value of center, left, right, up, down
vec3 normal;  // normal vector in object coordinates

// for extra credits #7:
// When using key "j", color the surface using the JetColorMap function, in the vertex shader. 
// Speficially, change the grayscale color x to JetColorMap(x).
void JetColorMap(float x, inout vec3 jetColor)  // have to addd inout qualifier because copy in and copy out
{
  float a; // alpha

  if (x < 0)
  {
    jetColor[0] = 0;
    jetColor[1] = 0;
    jetColor[2] = 0;
    return;
  }
  else if (x < 0.125) 
  {
    a = x / 0.125;
    jetColor[0] = 0;
    jetColor[1] = 0;
    jetColor[2] = 0.5 + 0.5 * a;
    return;
  }
  else if (x < 0.375)
  {
    a = (x - 0.125) / 0.25;
    jetColor[0] = 0;
    jetColor[1] = a;
    jetColor[2] = 1;
    return;
  }
  else if (x < 0.625)
  {
    a = (x - 0.375) / 0.25;
    jetColor[0] = a;
    jetColor[1] = 1;
    jetColor[2] = 1 - a;
    return;
  }
  else if (x < 0.875)
  {
    a = (x - 0.625) / 0.25;
    jetColor[0] = 1;
    jetColor[1] = 1 - a;
    jetColor[2] = 0;
    return;
  }
  else if (x <= 1.0)
  {
    a = (x - 0.875) / 0.125;
    jetColor[0] = 1 - 0.5 * a;
    jetColor[1] = 0;
    jetColor[2] = 0;
    return;
  }
  else
  {
    jetColor[0] = 1;
    jetColor[1] = 1;
    jetColor[2] = 1;
    return;
  }
}



void main()
{ // smooth triangle mode 
  if (controlRenderingMode == 4) {
    // compute the transformed and projected vertex position (into gl_Position, which is a predefined output variable in GLSL, 
    // that represents the transformed and projected position of a vertex in homogenous clip coordinates.  
    // In smooth mode, the vertex position is changed to the average position of itself and the four neighboring vertices. 
    // Component-wise vector operation.
    smoothPosition = (positionCenter + positionLeft + positionRight + positionDown + positionUp) / 5.0;    
    
    // TODO: extra credit #1 - too hard T.T
    if (lightingMode == 0) {
    }

    // Furthermore, change the vertex color (into col) following the formula given in hw1:
    float grayscale = pow(smoothPosition.y, exponent);  
    if (jetColorMapMode == 0){
      // IMPORTANT: col.rgb is a vec3, therefore cannot assign to scalar value. 
      // NO broadcasting in C++, unlike Python!
      if (numBytesPerPixel == 1) {  // grayscale input image
        if (secondImageMode == 0 || secondImageValid == 0) {  // core credit 
          col.r = grayscale;  
          col.g = grayscale; 
          col.b = grayscale; 
        } else {  // extra credit 5: color with second image
          col.r = grayscale * colorFromSecondImage.r;
          col.g = grayscale * colorFromSecondImage.g;
          col.b = grayscale * colorFromSecondImage.b;
        }
         
      } else if (numBytesPerPixel == 3) {  // extra credit: RGB input image
        if (secondImageMode == 0 || secondImageValid == 0) {  // based on color heightmap info
          col.r = color.r * grayscale;
          col.g = color.g * grayscale;
          col.b = color.b * grayscale;
        } else { // extra credit 5: based on second image supplied by user
          col.r = colorFromSecondImage.r * grayscale;
          col.g = colorFromSecondImage.g * grayscale;
          col.b = colorFromSecondImage.b * grayscale;
        } 
      }   
    } else { // TODO: extra creidt #7: Jet Color Map Mode
      vec3 jetColor;
      JetColorMap(grayscale, jetColor);
      col.rgb = jetColor;
    }

    // alpha value still = 1 
    col.a = color.a;  
    
    // Furthermore, change the vertex height based on scale and exponent input variables provided from the CPU.
    smoothPosition.y = scale * pow(smoothPosition.y, exponent);
    
    // Lastly, transform the resulting vertex position with the modelview and projection matrix as usual.
    gl_Position = projectionMatrix * modelViewMatrix * vec4(smoothPosition, 1.0f);
    
  } else {  // non-smooth mode
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0f);

      if (layerMode == 0  || controlRenderingMode != 2) {  
        // either not in layering mode, or objects to be rendered is not lines
        if (secondImageMode == 0 || secondImageValid == 0)  {  // core credit
          col = color;
        } else {  // extra credit 5: based on second image supplied by user
        col = colorFromSecondImage;
        }  
      } else  { 
        // In layer mode, and object to render is the mesh line that is layered on top of the solid. 
        // Hardcode line mesh color as blue. 
          col = color;
          col.b = 1.0;
      }
  }

}

