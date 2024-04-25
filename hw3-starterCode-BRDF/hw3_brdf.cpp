/* **************************
 * CSCI 420
 * Assignment 3 Raytracer
 * Name: Ziwei Duan
 * *************************
*/

#ifdef WIN32
  #include <windows.h>
#endif

#if defined(WIN32) || defined(linux)
  #include <GL/gl.h>
  #include <GL/glut.h>
#elif defined(__APPLE__)
  #include <OpenGL/gl.h>
  #include <GLUT/glut.h>
#endif

#if defined(WIN32) || defined(_WIN32)
#  define strcasecmp _stricmp
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#ifdef WIN32
  #define strcasecmp _stricmp
#endif

#include <imageIO.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <math.h>
#include <vector>
#include <array>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
std::random_device rd;
std::mt19937 eng;  // or eng(r()); for non-deterministic random number
std::uniform_real_distribution<double> distrib(0.0, 1.0 - 1e-8);
 



#define MAX_TRIANGLES 20000
#define MAX_SPHERES 100
#define MAX_LIGHTS 100

char * filename = NULL;

// The different display modes.
#define MODE_DISPLAY 1
#define MODE_JPEG 2

int mode = MODE_DISPLAY;
bool anti_aliasing = false;   // extra credit: turn on with --anti-aliasing=true in the command
int anti_aliasing_sample_size = 5;  // number of points sampled for each pixel, i.e. number of camera rays shoot for each pixel   
int brdf_sample_size = 15;  // extra credit

// While solving the homework, it is useful to make the below values smaller for debugging purposes.
// The still images that you need to submit with the homework should be at the below resolution (640x480).
// However, for your own purposes, after you have solved the homework, you can increase those values to obtain higher-resolution images.
#define WIDTH 640 //640 
#define HEIGHT 480  // 480
float aspect_ratio = 1.0 * WIDTH / HEIGHT;  // 1.0 * !!!! Otherwise int / int returns int (not float!)

// The field of view of the camera, in degrees. Convert to radius
float pi = 3.141592;
float fov = (40.0 / 180.0) * pi;
glm::vec3 camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
float z = -1.0; 
float zero_threshold = 0.0001;
float inf = 1000000.0;
glm::vec3 background_color(0.0, 0.0, 0.0);

// Buffer to store the image when saving it to a JPEG.
unsigned char buffer[HEIGHT][WIDTH][3];

struct Vertex
{
  double position[3];
  double normal[3];

  double color_diffuse[3];
  double roughness;
  double metallic;
};

struct Triangle
{
  Vertex v[3];
};

struct Sphere
{
  double position[3];
  double radius;

  double color_diffuse[3];
  double roughness;
  double metallic;
};

struct Light
{
  double position[3];
  double normal[3];
  double color[3];
  double p[4][3];
};

std::vector<Triangle> triangles;
std::vector<Sphere> spheres;
std::vector<Light> lights;
double F0[3];
double ambient_light[3];

#define ASERT(cond)                                                      \
  do {                                                                   \
    if ((cond) == false) {                                               \
      std::cerr << #cond << " failed at line " << __LINE__ << std::endl; \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

int num_triangles = 0;
int num_spheres = 0;
int num_lights = 0;

void plot_pixel_display(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel_jpeg(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel(int x,int y,unsigned char r,unsigned char g,unsigned char b);

// Helper function: for BRDF, positive characteristic function (which equals one if t > 0 and zero if t <= 0)
float stepResponse(float t) {
  if (t > zero_threshold) { return 1.0;
  } else {return zero_threshold; }
}
// Helper function 
float computeTriangleArea(glm::vec3 A, glm::vec3 B, glm::vec3 C) {
  glm::vec3 AB = B - A; 
  glm::vec3 AC = C - A; 
  return glm::length(glm::cross(AB, AC)) / 2;
}

// Helper function: compute the rectangular area of light. 
float computeLightArea(Light l) {
  glm::vec3 lightPositions[4];
  for (int i = 0; i < 4; i++) {  // convert light position to glm::vec3 type
    lightPositions[i] = glm::vec3(static_cast<float>(l.p[i][0]), static_cast<float>(l.p[i][1]), static_cast<float>(l.p[i][2]));
  }  
  return 2.0 * glm::length(computeTriangleArea(lightPositions[0], lightPositions[1], lightPositions[2]));
}
// Helper function: Halton Sequence 
// Goal: generate 2D sample points that are simultaneously well distributed over [0,1] and [0,1]
float halton(int index, int base) {
    float result = 0.0;
    float f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i /= base;
        f /= base;
    }
    return result;
}
// Helper function: 
// Input: Number of points in the 2D Halton sequence
void getHaltonSequence2D (int numPoints, float * xOffset, float * yOffset) {
    // printf("First %d points of the Halton sequence in 2D: \n", numPoints);
    for (int i = 0; i < numPoints; i++) {
        xOffset[i] = halton(i, 2);  // Base 2 for the first dimension
        yOffset[i] = halton(i, 3);  // Base 3 for the second dimension
        // printf("(%f, %f), \n", xOffset[i],  yOffset[i]);
    }
}

// Helper function: randomly sample a light source
void sampleLightPosition(glm::vec3 & lightPosition, glm::vec3 & lightColor, glm::vec3 & lightNormal, float & lightArea) {
    double U3 = distrib(eng);
    double U2 = distrib(eng);
    double U1 = distrib(eng);
    int sampledLightID = (int)std::min((int)(num_lights * U1), num_lights - 1);
    Light light = lights[sampledLightID];
    // read in corners of the area light and convert them into glm::vec3
    glm::vec3 p[4]; 
    for (int i = 0; i < 4; i++) {
      p[i] = glm::vec3(static_cast<float>(light.p[i][0]), static_cast<float>(light.p[i][1]), static_cast<float>(light.p[i][2]));
    }
    // sample light position point 
    for (int d = 0; d < 3; d++) {
      lightPosition[d] = (1- U2) * (p[0][d] * (1- U3) + p[1][d] * U3) + U2 * (p[2][d] * (1- U3) + p[3][d] * U3);
    }
    // fill in light color, light normal (which is uniform of light position sampled)
    lightColor = glm::vec3(static_cast<float>(light.color[0]), static_cast<float>(light.color[1]), static_cast<float>(light.color[2]));
    lightNormal = glm::vec3(static_cast<float>(light.normal[0]), static_cast<float>(light.normal[1]), static_cast<float>(light.normal[2]));
    lightArea = computeLightArea(light);

}







// Helper class 
class Ray{ 
  public:

    // Constructor: generate ("cast") a ray from origin parameter, in the direction of the direction parameter.  
    Ray(glm::vec3 originParm, glm::vec3 directionParm) : origin(originParm){
      direction = glm::normalize(directionParm);
    }
    
    // Calculate position of a point on the ray (Step 1 of HW3)
    glm::vec3 getPointPosition(float t) {
      if (t < 0) { return origin;  // t >= 0
      } else { return origin + t * direction; }
    } 

    // Find out the closest intersection of a ray cast from the camera to the given sphere (Step 2 of HW3),
    // Return true if there is at least 1 intersection, false otherwise
    // Only when return true, will we update value of the input intersection pointer to the closest intersection point. Otherwise = inf far.
    // Also we will update the unit normal of the closest intersection.
    bool getSphereIntersection(Sphere sphere, glm::vec3 * intersection, glm::vec3 * normalOfIntersection) {
      // initialize intersetion value to infinitly far
      *intersection = glm::vec3(inf, inf, inf);
      // convert double value of sphere center position to float and initialize glm::vec3
      glm::vec3 sphereCenter(static_cast<float>(sphere.position[0]), 
                             static_cast<float>(sphere.position[1]),
                             static_cast<float>(sphere.position[2])); 
      
      // Make sure ray shoots from outside of sphere 
      if (glm::length(origin - sphereCenter) - sphere.radius < zero_threshold) {
        // // printf("Ray shoots form inside/on the sphere, therefore no intersection.\n");
        return false; 
      }
      // Using the formula from lecture 16, slide 6
      float b = 2.0 * glm::dot(direction, origin - sphereCenter);
      float c = pow(glm::length(origin - sphereCenter), 2) - pow(sphere.radius, 2); 
      float condition = pow(b, 2) - 4.0 * c; 
      float t; 
      if (condition < -zero_threshold) {
        // printf("Condition = %f < 0, therefore no intersection.\n", condition);
        return false; }  // no intersection
      else if (condition < zero_threshold) {
        // printf("Condition = %f, therefore potentially 1 intersection.\n", condition);
        t = -b / 2.0;}  // condition within epson threshold to 0, therefore 1 intersection, i.e. ray is a tangent of sphere
      else {
        // printf("direction = [%f, %f, %f], origin - sphereCenter = [%f, %f, %f].\n", direction.x, direction.y, direction.z, (origin - sphereCenter).x, (origin - sphereCenter).y, (origin - sphereCenter).z);
        // printf("b = %f, c = %f, Condition = %f, therefore potentially 2 intersections.\n", b, c, condition);
        t = (-b - sqrt(condition)) / 2.0;  // NOTE: t0, t1 can only be both positive or both negative, because the ray is shoot from OUTSIDE of the sphere. 
      }

      // Make sure  t > 0 in order to intersect with ray
      if (t <= zero_threshold) {  // use epsilon instead of 0 to anti-aliasing
        // printf("t < 0, no intersection. \n");
        return false; } 
      else {
        // printf("intersection at [%f, %f, %f].\n", getPointPosition(t).x, getPointPosition(t).y, getPointPosition(t).z);
        *intersection = getPointPosition(t); 
        *normalOfIntersection = glm::normalize(*intersection - sphereCenter);
        if (abs(glm::length(*normalOfIntersection) - 1.0f) > zero_threshold) {
          // printf("!!!!!!! ERROR: normal of ray's intersection with the sphere = %f, is NOT normalized. !!!!!!!!! \n", glm::length(*normalOfIntersection));
        }
        return true;
      }
    }


    // Find out the intersection of a ray cast from the camera to the given triangle (Step 2 of HW3),
    // Return true if there exists an intersection (For the case ray intersecting with a triangle, there is always 0 or 1 intersection), false otherwise. 
    // Only when return true, will we update value of the input intersection pointer. Otherwise = inf far. 
    // Also we use Barycentric coordinates to interpolate unit normal, diffuse, specular and shininess coefficient of the intersection point inside the triangle. 
    bool getTriangleIntersection(Triangle triangle, glm::vec3 * intersection, glm::vec3 * normalOfIntersection, 
                                double colorDiffuseOfIntersection[3], double & roughnessOfIntersection, double & metallicOfIntersection) {
      *intersection = glm::vec3(inf, inf, inf);
      // 1) Find out the implicit form of the plane (ax + by + cz + d = 0) that contains the triangle 
      // First compute vector AB and AC
      Vertex vertexA = triangle.v[0];
      Vertex vertexB = triangle.v[1];
      Vertex vertexC = triangle.v[2];
      glm::vec3 positionA = glm::vec3(vertexA.position[0], vertexA.position[1], vertexA.position[2]);
      glm::vec3 positionB = glm::vec3(vertexB.position[0], vertexB.position[1], vertexB.position[2]);
      glm::vec3 positionC = glm::vec3(vertexC.position[0], vertexC.position[1], vertexC.position[2]);
      glm::vec3 AB = positionB - positionA;
      glm::vec3 AC = positionC - positionA; 
      
      // Then compute unit normal of the triangle (a, b, c) will give coefficients of the plane's implicit function ax + by + cz + d = 0
      glm::vec3 unitNormal = glm::normalize(glm::cross(AB, AC)); 
      
      // We can now determine whether there exists an intersection point. 
      // If the ray is orthorgnol to the normal, then the ray is parralle to the plane and has no intersection. 
      float normalDotDirection = glm::dot(unitNormal, direction);
      if (abs(normalDotDirection) <= zero_threshold) { 
        // printf("Ray is parallel to the triangle, therefore no intersection.\n");
        return false;
      }

      // To optimize, only compute coefficient d of the plane when there exists an intersection. 
      // Given any of the vertices A, B, C is on the plane and therefore satisfies the implicit function, we can compute coeff d. 
      float d  = - glm::dot(unitNormal, positionA);
      // printf("Implicit function of the plane containing the triangle has coefficient a = %f, b = %f, c = %f, d = %f \n", unitNormal.x, unitNormal.y, unitNormal.z, d);

      // 2) Compute the intersection point by figuring out its t (used in the parametric form of the ray) 
      float t = - (glm::dot(unitNormal, origin) + d) / normalDotDirection;
      // If t <= 0 then the intersection is behind ray origin. i.e. there is no intersection. 
      if (t <= zero_threshold) {  // use epsilon to anti-aliasing
        // printf("t = %f <= 0, therefore no intersection. \n", t);
        return false; 
      }
      glm::vec3 positionP = getPointPosition(t);
      
      // 3) Deterine whether the intersection point is inside the triangle 
      glm::vec3 PB = positionB - positionP; 
      glm::vec3 PC = positionC - positionP; 
      // float alpha = abs(glm::cross(PB, PC).z / glm::cross(AB, AC).z);
      float alpha = glm::length(glm::cross(PB, PC)) / glm::length(glm::cross(AB, AC));

      glm::vec3 PA = positionA - positionP; 
      glm::vec3 BA = positionA - positionB; 
      glm::vec3 BC = positionC - positionB; 
      // float beta = abs(glm::cross(PA, PC).z / glm::cross(BA, BC).z);
      float beta = glm::length(glm::cross(PA, PC))/ glm::length(glm::cross(BA, BC));
      float gamma = glm::length(glm::cross(PA, PB)) / glm::length(glm::cross(-AC, -BC));
      // printf("Barycentric coordinates: alpha = %f, beta = %f, gamma = %f \n", alpha, beta, gamma);
      
      if (alpha < - zero_threshold || alpha > 1.0 + zero_threshold ||
          beta < - zero_threshold || beta > 1.0 + zero_threshold || 
          gamma < - zero_threshold || gamma > 1.0 + zero_threshold || 
          abs(alpha + beta + gamma - 1.0) > zero_threshold) {
        // printf("Fail the IFF condition that alpha, beta, gamma all between [0, 1] and sum = 1. Therefore no intersection inside the triangle\n");
        return false;  // intersection point P is outside of triangle
      } else {  // intersection point P is inside triangle. 
        *intersection = positionP;
        // Interpolate normal using Barycentric coordinates
        glm::vec3 normalVertexA = glm::vec3(vertexA.normal[0], vertexA.normal[1], vertexA.normal[2]);
        glm::vec3 normalVertexB = glm::vec3(vertexB.normal[0], vertexB.normal[1], vertexB.normal[2]);
        glm::vec3 normalVertexC = glm::vec3(vertexC.normal[0], vertexC.normal[1], vertexC.normal[2]);
        *normalOfIntersection = glm::normalize(alpha * normalVertexA + beta * normalVertexB + gamma * normalVertexC);
        if (abs(glm::length(*normalOfIntersection) - 1) > zero_threshold) {
          // printf("!!!!!!! ERROR: normal of ray's intersection = %f with the triangle is NOT normalized. !!!!!!!!! \n", glm::length(*normalOfIntersection) );
        }
        // Interpolate diffuse coeff, specular coeff, shininess coeff for intersection point P.
        metallicOfIntersection = alpha * vertexA.metallic + beta * vertexB.metallic + gamma * vertexC.metallic;
        roughnessOfIntersection = alpha * vertexA.roughness + beta * vertexB.roughness + gamma * vertexC.roughness;
        for (int channel = 0; channel < 3; channel++) { // interpolate for each color channel
          colorDiffuseOfIntersection[channel] = alpha * vertexA.color_diffuse[channel] + beta * vertexB.color_diffuse[channel] + gamma * vertexC.color_diffuse[channel];
        }
        // printf("Intersection inside triangle! \n");
        return true;
      }  // intersection point P is inside the triangle. i.e. ray intersects with the triangle
    }


    // Find out the closest intersection point and its distance from ray source among ALL objects in the scene.
    // (there are only triangles and spheres primitives in the scene. )
    // Return true if intersecting with at least 1 object in the scene, false otherwise. 
    bool getClosestIntersection(glm::vec3 * closestIntersection, glm::vec3 * normalOfClosestIntersection, 
                                double colorDiffuseOfClosestIntersection[3], double & roughnessOfClosestIntersesction, double & metallicOfClosestIntersection) {
      bool intersectionFlag = false;  // switch to on as long as there is 1 intersection
      *closestIntersection = glm::vec3(inf, inf, inf);  // Initialize closest intersection to be infinitely far. 
      float closestDistance = glm::length(*closestIntersection - camera_position); 
      // must initialize. Otherwise dereferencing a null pointer inside the function will lead to seg fault. 
      glm::vec3 candidateIntersection(inf, inf, inf);  
      glm::vec3 candidateNormal(0.0, 0.0, 0.0);
      float candidateDistance;
      double candidateRoughness, candidateMetallic; 
      double candidateColorDiffuse[3];

      // Loop through all spheres in the scene to find out the closest intersection of all spheres with this ray.
      for (int i = 0; i < num_spheres; i++) {
        // printf("Check intersection with sphere %d:\n", i);
        if (getSphereIntersection(spheres[i], &candidateIntersection, &candidateNormal)) {   // there is intersection
          intersectionFlag = true; 
          // Determine whether this intersection is < the running closest intersection
          candidateDistance = glm::length(candidateIntersection - camera_position);  
          if (candidateDistance < closestDistance) {  // update "closest"
            closestDistance = candidateDistance;
            *closestIntersection = candidateIntersection;
            *normalOfClosestIntersection = candidateNormal;
            // printf("Updating closest intersection = [%f, %f, %f], normal = [%f, %f, %f].\n", 
            //                (*closestIntersection).x, (*closestIntersection).y, (*closestIntersection).z, 
            //                (*normalOfClosestIntersection).x, (*normalOfClosestIntersection).y, (*normalOfClosestIntersection).z);
            for (int channel = 0; channel < 3; channel++) {
              colorDiffuseOfClosestIntersection[channel] = spheres[i].color_diffuse[channel];                
            }
            roughnessOfClosestIntersesction = spheres[i].roughness;
            metallicOfClosestIntersection = spheres[i].metallic;
            // printf("Diffuse = [%f, %f, %f], specular = [%f, %f, %f], shininess = %f.\n", 
            //        colorDiffuseOfClosestIntersection[0], colorDiffuseOfClosestIntersection[1], colorDiffuseOfClosestIntersection[2],
            //        colorSpecularOfClosestIntersection[0], colorSpecularOfClosestIntersection[1], colorSpecularOfClosestIntersection[2], 
            //        shininessOfClosestIntersection);
          }
        } // No intersection and therefore do NOT update closesetDistance nor closestIntersection 
      }  

      // Loop through all triangles in the scene to find out the closest intersection of all objects in the scene
      for (int i = 0; i < num_triangles; i++) {
        // printf("Check intersection with triangle %d: \n", i);
        if (getTriangleIntersection(triangles[i], &candidateIntersection, &candidateNormal, candidateColorDiffuse, candidateRoughness, candidateMetallic)) {  // intersect
          intersectionFlag = true;
          // Determine whether this intersection is < the running closest intersection
          candidateDistance = glm::length(candidateIntersection - camera_position);  
          if (candidateDistance < closestDistance) {  // update "closest"
            closestDistance = candidateDistance;
            *closestIntersection = candidateIntersection;
            *normalOfClosestIntersection = candidateNormal;
            // printf("Updating closest intersection = [%f, %f, %f], normal = [%f, %f, %f].\n", 
            //                (*closestIntersection).x, (*closestIntersection).y, (*closestIntersection).z, 
            //                (*normalOfClosestIntersection).x, (*normalOfClosestIntersection).y, (*normalOfClosestIntersection).z);
            for (int channel = 0; channel < 3; channel++) {
              colorDiffuseOfClosestIntersection[channel] = candidateColorDiffuse[channel];                
            }
            roughnessOfClosestIntersesction = candidateRoughness;
            metallicOfClosestIntersection = candidateMetallic;
            // printf("Diffuse = [%f, %f, %f], specular = [%f, %f, %f], shininess = %f.\n", 
            //        colorDiffuseOfClosestIntersection[0], colorDiffuseOfClosestIntersection[1], colorDiffuseOfClosestIntersection[2],
            //        colorSpecularOfClosestIntersection[0], colorSpecularOfClosestIntersection[1], colorSpecularOfClosestIntersection[2], 
            //        shininessOfClosestIntersection);            
          }  
        } // otherwise, no intersection, and therefore do NOT update closesetDistance nor closestIntersection 
      }

      return intersectionFlag;  
    }
    
    glm::vec3 getOrigin() {
      return origin;
    }
    glm::vec3 getDirection() {
      return direction;
    }


  private:
    glm::vec3 origin;
    glm::vec3 direction;

};

// Helper function: compute cosine square of the angle between vector a and b
float computeCosineSquare(glm::vec3 a, glm::vec3 b) {
  float cos = glm::dot(a, b) / (glm::length(a) * glm::length(b));
  return pow(cos, 2);
} 
// Helper function: compute tangent square of the angle between a and b
float computeTangentSquare(glm::vec3 a, glm::vec3 b) {
  float cosSquare = computeCosineSquare(a, b);
  return (1.0 - cosSquare) / cosSquare;
}
// Helper function: compute G1(v, m, n, rough)
float computeG1(glm::vec3 v, glm::vec3 m, glm::vec3 n, double rough) {
  float alpha = pow(static_cast<float>(rough), 2);
  float tanSquareVandN = computeTangentSquare(v, n);
  float denominator = 1.0 + sqrt(1.0 + pow(alpha, 2) * tanSquareVandN);
  float numerator = 2.0 * stepResponse(glm::dot(v, m) * glm::dot(v, n));

  return numerator / denominator;
}
// Helper function: compute D(m, n, rough)
float computeD(glm::vec3 m, glm::vec3 n, double rough) {
  float alphaSquare = pow(static_cast<float>(rough), 4);
  float numerator = alphaSquare * stepResponse(glm::dot(m, n));
  
  float cosSquareMandN = computeCosineSquare(m, n);
  float tanSquareMandN = computeTangentSquare(m, n);
  float denominator = pi * pow(cosSquareMandN, 2) * pow(alphaSquare + tanSquareMandN,2);

  // Found out where the aliasing (white dot around the boundaries) come from).
  // if (numerator / denominator > 2500) {printf("cos^4 = %f, alpha^2 = %f, tanSquare = %f\n", pow(cosSquareMandN, 2), alphaSquare, tanSquareMandN);}
  return numerator/ denominator;
}

// Helper function that break down BRDF, evaluate the BRDF:
glm::vec3 computeBRDF(glm::vec3 incomingLight, glm::vec3 normalOfClosestIntersection, glm::vec3 viewVector,
                      double diffuseCoefficient[3], double roughnessCoefficient, double metallicCoefficient) {
  // h
  glm::vec3 h;
  if (glm::dot(incomingLight, normalOfClosestIntersection) < - zero_threshold) {  // < 0
    h = - glm::normalize(incomingLight + viewVector);
  } else if (glm::dot(incomingLight, normalOfClosestIntersection) < zero_threshold) {// == 0
    h = glm::vec3(zero_threshold, zero_threshold, zero_threshold); 
  } else {
    h = glm::normalize(incomingLight + viewVector);
  }
  // F0 (base reflection)   
  glm::vec3 baseReflection(static_cast<float>(F0[0]), static_cast<float>(F0[1]), static_cast<float>(F0[2]));
  // F
  glm::vec3 F;
  for (int channel = 0; channel < 3; channel++) {
    F[channel] = baseReflection[channel] + (1.0f - baseReflection[channel]) * pow(1 - glm::dot(viewVector, h), 5);
  } 
  // FD90, scalar
  float FD90 = 2 * pow(glm::dot(h, incomingLight), 2) * roughnessCoefficient + 0.5;
 
  // G
  float G = computeG1(incomingLight, h, normalOfClosestIntersection, roughnessCoefficient) * computeG1(viewVector, h, normalOfClosestIntersection, roughnessCoefficient);
  
  // D
  float D = computeD(h, normalOfClosestIntersection, roughnessCoefficient);

  // f_s, formula gives a vector
  float fsScalar = G *  D / (4.0 * abs(glm::dot(incomingLight, normalOfClosestIntersection)) * abs(glm::dot(viewVector, normalOfClosestIntersection)));
 
  glm::vec3 fsVector = F * glm::vec3(fsScalar, fsScalar, fsScalar);

  // diffuse f_d formula gives a scalar. You can make a Vector3 with each component equal to this scalar. @264 Piazza post
  float fdScalar = (1.0 + (FD90 - 1.0) * pow(1 - glm::dot(incomingLight, normalOfClosestIntersection), 5))  
                            * (1.0 + (FD90 - 1.0) * pow(1 - glm::dot(viewVector, normalOfClosestIntersection), 5)) 
                            * (1.0 - metallicCoefficient)
                            / pi;                          
  glm::vec3 fdVector(fdScalar, fdScalar, fdScalar);

  // albedo is the base reflection, which is the base color of the object 
  glm::vec3 albedo(static_cast<float>(diffuseCoefficient[0]), static_cast<float>(diffuseCoefficient[1]), static_cast<float>(diffuseCoefficient[2]));
  printf("fsVector = %f, %f, %f, fdVector = %f, %f, %f \n", fsVector[0], fsVector[1], fsVector[2], fdVector[0], fdVector[1], fdVector[2]);
  return (fsVector + fdVector) * albedo;

}

// Helper Function that obtains probability density function (pdf) of w_i (helps break down BRDF)
float computePDF(glm::vec3 intersection, glm::vec3 lightPosition, glm::vec3 lightNormal, float lightArea, glm::vec3 incomingLight) {
  float LightIntersectionDistance = glm::length(lightPosition - intersection);
  glm::vec3 unitLightNormal = glm::normalize(lightNormal);
  glm::vec3 unitIncomingLight = glm::normalize(incomingLight);
  float cosLightNormalAndIncomingLight = glm::dot(unitLightNormal, unitIncomingLight);
  return pow(LightIntersectionDistance, 2) / (lightArea * abs(cosLightNormalAndIncomingLight));
}

// Helper function 
// When a light is NOT blocked, compute its contribution to the color of intersection point (from where the shadow ray is shoot) 
glm::vec3 applyBRDFShading(Ray shadowRay, glm::vec3 lightPosition, glm::vec3 lightColor, glm::vec3 lightNormal, float lightArea,
                            glm::vec3 closestIntersection, glm::vec3 normalOfClosestIntersection, 
                            double diffuseCoefficient[3], double roughnessCoefficient, double metallicCoefficient) {
  glm::vec3 incomingLight = shadowRay.getPointPosition(1.0) - closestIntersection;  // then compute incoming light (NOTE it's a unit vector)
  
  // Compute the reflection light
  glm::vec3 reflectionLight = 2 * glm::dot(incomingLight, normalOfClosestIntersection) * normalOfClosestIntersection - incomingLight;
  glm::vec3 viewVector = glm::normalize(camera_position - closestIntersection);  // recall view vector is from intersection point, towards camera eye
  // printf("Normal n = [%f, %f, %f], \n", normalOfClosestIntersection.x, normalOfClosestIntersection.y, normalOfClosestIntersection.z);
  // printf("Incoming light l = [%f, %f, %f], \n", incomingLight.x, incomingLight.y, incomingLight.z);
  // printf("Reflection light r = [%f, %f, %f], \n", reflectionLight.x, reflectionLight.y, reflectionLight.z);
  // printf("view vector v = [%f, %f, %f]. \n", viewVector.x, viewVector.y, viewVector.z);
  
  // Compute Light Color Le. Since when light is blocked, contribution of this light = 0, so is Le. 
  // Here we are applying BDRF, which implies we have checked that light is NOT blocked.  
  glm::vec3 Le; 
  if (glm::dot(incomingLight, normalOfClosestIntersection) < zero_threshold) {
    Le = glm::vec3(zero_threshold, zero_threshold, zero_threshold);
  } else {
    Le = lightColor;
  }
  // Compute component of BDRF 
  glm::vec3 BRDF = computeBRDF(incomingLight, normalOfClosestIntersection, viewVector, diffuseCoefficient, roughnessCoefficient, metallicCoefficient);

  // Compute w_i dot n
  float angleIncomingLightAndIntersectionNormal = std::max(glm::dot(incomingLight, normalOfClosestIntersection), zero_threshold);
  
  
  // Compute pdf
  float pdf = computePDF(closestIntersection, lightPosition, lightNormal, lightArea, incomingLight);

  // Apply BRDF formula to determine the color of the pixel with respect to that light
  glm::vec3 BRDFColor; 
  for (int channel = 0; channel < 3; channel++) {
    BRDFColor[channel] = Le[channel] * BRDF[channel] * angleIncomingLightAndIntersectionNormal / pdf;
  }
  // return glm::clamp(BRDFColor, zero_threshold, 1000.0f);
  return BRDFColor;
}


// Modularize camera ray casting given camera position and pixel position 
// return color obtained by this camera ray. 
glm::vec3 computeColorForCameraRay(glm::vec3 _pixelPosition) {
  // Initiate pixel color as background color. 
  glm::vec3 color = background_color; 
  
  // 1) Cast a ray from camera position, whcih is (0, 0, 0) in camera frame the pixel center, 
  Ray cameraRay = Ray(camera_position, _pixelPosition - camera_position);  
  // printf("Ray origin = [%f, %f, %f], direction = [%f, %f, %f]. \n", cameraRay.getOrigin().x, cameraRay.getOrigin().y, cameraRay.getOrigin().z, cameraRay.getDirection().x, cameraRay.getDirection().y, cameraRay.getDirection().z); 
  
  // Among ALL objects in the scene, find out the closest intersection of the backward ray shooting from the camera to the pixel. 
  glm::vec3 closestIntersection(inf, inf, inf);
  glm::vec3 normalOfClosestIntersection(0.0, 0.0, 0.0);
  double roughness, metallic; 
  double diffuseCoefficient[3];
  if (cameraRay.getClosestIntersection(&closestIntersection, &normalOfClosestIntersection, diffuseCoefficient, roughness, metallic)) {  
    // This is the case where backward ray hits some object in the scene, then its pixel color is determined by ALL the lights reflected by this object. 
    // we can "see" the object because of ambient light as a light source 
    color.r = static_cast<float>(ambient_light[0]); 
    color.g = static_cast<float>(ambient_light[1]);
    color.b = static_cast<float>(ambient_light[2]); 
    // printf("Camera ray hits an object! Start with ambient color = [%f, %f, %f].\n", color.r, color.g, color.b);

    // 2) Sampling the light source. Find out color contributed the sampled light sources by shooting shadow ray.
    // printf("\nStep 2: Cast shadow ray for each light:\n");
    glm::vec3 lightPosition, lightColor, lightNormal;
    float lightArea;
    sampleLightPosition(lightPosition, lightColor, lightNormal, lightArea);

    // cast a shadow ray from the closest intersection point to each light source
    Ray shadowRay = Ray(closestIntersection, lightPosition - closestIntersection);

    // Among ALL objects in the scene, find out the closest intersection of the shadow ray from intersection point to the light 
    glm::vec3 closestBlockageIntersection(inf, inf, inf);
    glm::vec3 normalOfClosestBlockageIntersection(0.0, 0.0, 0.0);
    double roughnessOfClosestBlockage, metallicOfClosestBlockage; 
    double diffuseCoefficientOfClosestBlockage[3];

    if (!shadowRay.getClosestIntersection(&closestBlockageIntersection, &normalOfClosestBlockageIntersection, diffuseCoefficientOfClosestBlockage, roughnessOfClosestBlockage, metallicOfClosestBlockage)) {
      // No objects blocking between intersection surface point and the light 
      // printf("Light %d (pos = [%f, %f, %f] is NOT blocked. Intersection point existing color = [%f, %f, %f]. \n", 
      //      i, lightPosition.x, lightPosition.y, lightPosition.z, lightColor.r, lightColor.g, lightColor.b);
      // Apply BRDF shading for the unblocked light onto the intersection point 
      glm::vec3 BRDFShadingColor = applyBRDFShading(shadowRay, lightPosition, lightColor, lightNormal, lightArea,
                          closestIntersection, normalOfClosestIntersection, 
                          diffuseCoefficient, roughness, metallic); 
      color = color + BRDFShadingColor;  // add to exisitng color of the intersection point                                            
  
    } else {  // Shadow ray hit some object. 
      // ++++++++++++++ Still need to check whether the object is IN BETWEEN light and the intersection! ++++++++++++++++++++++++++
      // ++++++++++++++ If the object hit is behind the light, then it's NOT an obstable, and the intersection is NOT in shadow! ++++++++++++++
      float distanceFromIntersectionToLight = (lightPosition.x - closestIntersection.x) / shadowRay.getDirection().x;
      float distanceFromIntersectionToBlockage = (closestBlockageIntersection.x - closestIntersection.x) / shadowRay.getDirection().x; 
      if (distanceFromIntersectionToLight - distanceFromIntersectionToBlockage > zero_threshold) {  
        // Obstacle is IN BETWEEN intersection and Light, therefore blocking the light.
        // No color contribution w.r.t this light. 
        // printf("Lights %d is blocked. No contribution from this light. \n", i);
      } else {
        // Since the hit object is "behind" the light (i.e. Object - Light - Intersection), it does NOT block the light
        // printf("Because the 'obstable' pos = [%f, %f, %f], it's BEHIND Light %d (pos = [%f, %f, %f]) and does NOT block the light.\n Intersection point existing color = [%f, %f, %f]. \n", 
        //    closestBlockageIntersection.x, closestBlockageIntersection.y, closestBlockageIntersection.z,  
        //    i, lightPosition.x, lightPosition.y, lightPosition.z, lightColor.r, lightColor.g, lightColor.b);
        // Apply BRDF shading for the unblocked light onto the intersection point 
        glm::vec3 BRDFShadingColor = applyBRDFShading(shadowRay, lightPosition, lightColor, lightNormal, lightArea,
                          closestIntersection, normalOfClosestIntersection, 
                          diffuseCoefficient, roughness, metallic); 
        color = color + BRDFShadingColor;  // add to exisitng color of the intersection point    
      } 
    }
    // printf("After shadow ray for light %d, color = [%f, %f, %f] \n",  i, color.r, color.g, color.b);
  } // backward ray does not hit any object in the scene, therefore pixel color is the background color (no modification to initial color)
  
 
  return glm::clamp(color, zero_threshold, 1000.0f);
}

void draw_scene()
{ 
  // Calculate boundaries of camera frame
  float yMax = tan(fov / 2);   // printf("yMax = %f\n", yMax);
  float xMax = yMax * aspect_ratio;  // printf("xMax = %f\n", xMax);
  float pixelWidth = xMax * 2 / WIDTH;  // width of each pixel box
  float pixelHeight = yMax * 2 / HEIGHT;  // height of each pixel box
  // start from the center of pixel at the LOWER LEFT corner of the image plane, 
  // traversing/filling up the image from bottom up, from left to right.
  glm::vec3 pixelBottomLeftPos = glm::vec3(-xMax, -yMax, z);
  
  for(unsigned int x=0; x<WIDTH; x++) 
  {
    glPointSize(2.0);  
    // Do not worry about this usage of OpenGL. This is here just so that we can draw the pixels to the screen,
    // after their R,G,B colors were determined by the ray tracer.
    glBegin(GL_POINTS);

    for(unsigned int y=0; y<HEIGHT; y++) {  // For each pixel on the image plane, determine its rgb color [0, 255]
      // printf("--------------------------\n");
      // printf("Step 1: Cast a ray from camera to pixel [%d, %d] to check for intersection with objects in the scene.\n ", x, y);

      glm::vec3 color; 
      // Sample points inside the pixel box, and shoot camera ray for every sampled pixel points on the image plane.  
      float xOffsets[brdf_sample_size], yOffsets[brdf_sample_size];  // x, y, are both [0, 1]
      getHaltonSequence2D(brdf_sample_size, xOffsets, yOffsets);
      for (int s = 0; s < brdf_sample_size; s++) {
        glm::vec3 samplePixelPos = pixelBottomLeftPos + glm::vec3(xOffsets[s] * pixelWidth, yOffsets[s] * pixelHeight, 0);
        glm::vec3 sampleColor = computeColorForCameraRay(samplePixelPos);
        
        color += sampleColor / glm::vec3(brdf_sample_size, brdf_sample_size, brdf_sample_size);   // weighted by sample size, and added to the pixel color 
      }

      for (int channel = 0; channel < 3; channel++) {
        // Tone mapping. 
        // Also you guarantee that the values of the final color are between 0 and 1, and preserve color better them clamping to 1
        color[channel] = color[channel] / (color[channel] + 1.0);  
      }

      // printf("Final pixel color floating point = [%f, %f, %f].\n", color.r, color.g, color.b);
      // Convert glm::vec3 color (floating point from 0 to 1) to integer R,G,B output (from 0 to 255).
      // Modify these R,G,B colors to the values computed by your ray tracer (If final color channel value > 1, then clamp to 1).
      unsigned char r, g, b;
      r = (int) (std::min(1.0f, color.r) * 255);  // NOTE!! (int) cast has higher precedence!!
      g = (int) (std::min(1.0f, color.g) * 255); 
      b = (int) (std::min(1.0f, color.b) * 255); 
      plot_pixel(x, y, r, g, b);    

      // move onto the next pixel above (scanning the image plane column by column) 
      pixelBottomLeftPos.y += pixelHeight;
    }
    // After scanning one column (all y on a fixed x) move to the next column of pixels 
    // by incrementing x coordinate, and resetting y coordinate to the bottom of image plane. 
    pixelBottomLeftPos.x += pixelWidth; 
    pixelBottomLeftPos.y = -yMax;
    glEnd();
    glFlush();
  } 
  
  printf("Ray tracing completed.\n"); 
  fflush(stdout);
}

void plot_pixel_display(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  glColor3f(((float)r) / 255.0f, ((float)g) / 255.0f, ((float)b) / 255.0f);
  glVertex2i(x,y);
}

void plot_pixel_jpeg(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  buffer[y][x][0] = r;
  buffer[y][x][1] = g;
  buffer[y][x][2] = b;
}

void plot_pixel(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  plot_pixel_display(x,y,r,g,b);
  if(mode == MODE_JPEG)
    plot_pixel_jpeg(x,y,r,g,b);
}

void save_jpg()
{
  printf("Saving JPEG file: %s\n", filename);

  ImageIO img(WIDTH, HEIGHT, 3, &buffer[0][0][0]);
  if (img.save(filename, ImageIO::FORMAT_JPEG) != ImageIO::OK)
    printf("Error in saving\n");
  else 
    printf("File saved successfully\n");
}

void parse_check(const char *expected, char *found)
{
  if (strcasecmp(expected, found)) {
    printf("Expected '%s ' found '%s '\n", expected, found);
    printf("Parse error, abnormal abortion\n");
    exit(1);
  }
}

void parse_doubles(FILE *file, const char *check, double p[3])
{
  char str[512];
  int ret = fscanf(file, "%s", str);
  ASERT(ret == 1);

  parse_check(check, str);

  ret = fscanf(file, "%lf %lf %lf", &p[0], &p[1], &p[2]);
  ASERT(ret == 3);

  printf("%s %lf %lf %lf\n", check, p[0], p[1], p[2]);
}

void parse_double(FILE *file, const char *check, double &r)
{
  char str[512];
  int ret = fscanf(file, "%s", str);
  ASERT(ret == 1);

  parse_check(check, str);

  ret = fscanf(file, "%lf", &r);
  ASERT(ret == 1);

  printf("%s %f\n", check, r);
}

void parse_rad(FILE *file, double *r)
{
  char str[100];
  fscanf(file,"%s",str);
  parse_check("rad:",str);
  fscanf(file,"%lf",r);
  printf("rad: %f\n",*r);
}

void parse_shi(FILE *file, double *shi)
{
  char s[100];
  fscanf(file,"%s",s);
  parse_check("shi:",s);
  fscanf(file,"%lf",shi);
  printf("shi: %f\n",*shi);
}

int loadScene(const char *filename)
{
  FILE *file = fopen(filename, "r");
  int number_of_objects;
  char type[50] = { 0 };
  Triangle t;
  Sphere s;
  Light l;

  int ret = fscanf(file, "%i", &number_of_objects);
  ASERT(ret == 1);

  printf("number of objects: %i\n", number_of_objects);

  parse_doubles(file, "amb:", ambient_light);

  parse_doubles(file, "f0:", F0);

  for (int i = 0; i < number_of_objects; i++) {
    int ret = fscanf(file, "%s\n", type);
    ASERT(ret == 1);

    // printf("%s\n", type);
    if (strcasecmp(type, "triangle") == 0) {
      printf("found triangle\n");
      for (int j = 0; j < 3; j++) {
        parse_doubles(file, "pos:", t.v[j].position);
        parse_doubles(file, "nor:", t.v[j].normal);
        parse_doubles(file, "dif:", t.v[j].color_diffuse);

        parse_double(file, "rou:", t.v[j].roughness);
        parse_double(file, "met:", t.v[j].metallic);
      }

      if ((int)triangles.size() == MAX_TRIANGLES) {
        printf("too many triangles, you should increase MAX_TRIANGLES!\n");
        exit(0);
      }

      triangles.push_back(t);
      num_triangles++;
    }
    else if (strcasecmp(type, "sphere") == 0) {
      printf("found sphere\n");

      parse_doubles(file, "pos:", s.position);
      parse_double(file, "rad:", s.radius);
      parse_doubles(file, "dif:", s.color_diffuse);

      parse_double(file, "rou:", s.roughness);
      parse_double(file, "met:", s.metallic);

      if ((int)spheres.size() == MAX_SPHERES) {
        printf("too many spheres, you should increase MAX_SPHERES!\n");
        exit(0);
      }

      spheres.push_back(s);
      num_spheres++;
    }
    else if (strcasecmp(type, "light") == 0) {
      printf("found light\n");
      parse_doubles(file, "p0:", l.p[0]);
      parse_doubles(file, "p1:", l.p[1]);
      parse_doubles(file, "p2:", l.p[2]);
      parse_doubles(file, "p3:", l.p[3]);

      parse_doubles(file, "pos:", l.position);
      parse_doubles(file, "nrm:", l.normal);
      parse_doubles(file, "col:", l.color);

      if ((int)lights.size() == MAX_LIGHTS) {
        printf("too many lights, you should increase MAX_LIGHTS!\n");
        exit(0);
      }
      lights.push_back(l);
      num_lights++;
    }
    else {
      printf("unknown type in scene description:\n%s\n", type);
      exit(0);
    }
  }
  printf("Number of spheres = %zu, number of triangles = %zu, number of lights = %zu. \n", spheres.size(), triangles.size(), lights.size());
  return 0;
}


void display()
{
}

void init()
{
  glMatrixMode(GL_PROJECTION);
  glOrtho(0,WIDTH,0,HEIGHT,1,-1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glClearColor(0,0,0,0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void idle()
{
  // Hack to make it only draw once.
  static int once=0;
  if(!once)
  {
    draw_scene();
    if(mode == MODE_JPEG)
      save_jpg();
  }
  once=1;
}

int main(int argc, char ** argv)
{
  if ((argc < 2) || (argc > 5))
  {  
    printf ("Usage: %s <input scenefile> [output jpegname]\n", argv[0]);
    exit(0);
  }
  if(argc >= 3)
  {
    mode = MODE_JPEG;
    filename = argv[2];
  }
  else if(argc == 2)
    mode = MODE_DISPLAY;


  // Parse extra command flag for extra credits
  printf("========== Parsing arguments:\n");
  std::string argString, argFlag, argValue; 
  size_t equalSignPos;  
  for (int c = 3; c <argc; c++) {
    argString = argv[c];
    if (argString.substr(0, 2) == "--") {
      // Find the position of the '=' character 
      equalSignPos = argString.find('=');
      if (equalSignPos != std::string::npos) {
        // extract the flag and value, which are given in the command line as --flag=value
        argFlag = argString.substr(2, equalSignPos - 2);   // substr (size_t pos = 0, size_t len = npos)
        argValue = argString.substr(equalSignPos + 1);
        printf("Flag: %s, Value: %s\n", argFlag.c_str(), argValue.c_str());
      }

      // After parsing out flag and its value, we make them known to the program:
      // --anti-aliasing=false|true
      if (argFlag == "anti-aliasing") {  
        if (argValue == "true") {anti_aliasing = true; }  // by default is fasle.
      }

      if (argFlag == "brdf_sample_size") {
        brdf_sample_size = std::stoi(argValue);
        printf("User input: brdf_sample_size = %d. \n", brdf_sample_size);
      }
 
    }
  }  



  glutInit(&argc,argv);
  loadScene(argv[1]);

  glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
  glutInitWindowPosition(0,0);
  glutInitWindowSize(WIDTH,HEIGHT);
  int window = glutCreateWindow("Ray Tracer");
  #ifdef __APPLE__
    // This is needed on recent Mac OS X versions to correctly display the window.
    glutReshapeWindow(WIDTH - 1, HEIGHT - 1);
  #endif
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  init();
  glutMainLoop();
}

