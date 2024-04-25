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
 



#define MAX_TRIANGLES 20000
#define MAX_SPHERES 100
#define MAX_LIGHTS 100
int MAX_RECURSION = 1;  // 0 means no reflection ray tracing, i.e. only primary ray (shadow rays) 

char * filename = NULL;

// The different display modes.
#define MODE_DISPLAY 1
#define MODE_JPEG 2

int mode = MODE_DISPLAY;
bool anti_aliasing = true;   // extra credit
int anti_aliasing_sample_size = 5;  // number of points sampled for each pixel, i.e. number of camera rays shoot for each pixel   

// While solving the homework, it is useful to make the below values smaller for debugging purposes.
// The still images that you need to submit with the homework should be at the below resolution (640x480).
// However, for your own purposes, after you have solved the homework, you can increase those values to obtain higher-resolution images.
#define WIDTH 640 //640 
#define HEIGHT 480  // 480
float aspect_ratio = 1.0 * WIDTH / HEIGHT;  // 1.0 * !!!! Otherwise int / int returns int (not float!)

// The field of view of the camera, in degrees. Convert to radius
float fov = (60.0 / 180.0) * 3.141592;
glm::vec3 camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
float z = -1.0; 
float zero_threshold = 0.0001;
float inf = 1000000.0;
glm::vec3 background_color(1.0, 1.0, 1.0);
float reflection_color_weight = 0.2;

// Buffer to store the image when saving it to a JPEG.
unsigned char buffer[HEIGHT][WIDTH][3];

struct Vertex
{
  double position[3];
  double color_diffuse[3];
  double color_specular[3];
  double normal[3];
  double shininess;
};

struct Triangle
{
  Vertex v[3];
};

struct Sphere
{
  double position[3];
  double color_diffuse[3];
  double color_specular[3];
  double shininess;
  double radius;
};

struct Light
{
  double position[3];
  double color[3];
};
Triangle triangles[MAX_TRIANGLES];
Sphere spheres[MAX_SPHERES];
Light lights[MAX_LIGHTS];
double ambient_light[3];

int num_triangles = 0;
int num_spheres = 0;
int num_lights = 0;

void plot_pixel_display(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel_jpeg(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel(int x,int y,unsigned char r,unsigned char g,unsigned char b);

// Helper function 
float computeTriangleArea(glm::vec3 A, glm::vec3 B, glm::vec3 C) {
  glm::vec3 AB = B - A; 
  glm::vec3 AC = C - A; 
  return glm::length(glm::cross(AB, AC)) / 2;
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
                                double colorDiffuseOfIntersection[3], double colorSpecularOfIntersection[3], double & shininessOfIntersection) {
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
        shininessOfIntersection = alpha * vertexA.shininess + beta * vertexB.shininess + gamma * vertexC.shininess;
        for (int channel = 0; channel < 3; channel++) { // interpolate for each color channel
          colorDiffuseOfIntersection[channel] = alpha * vertexA.color_diffuse[channel] + beta * vertexB.color_diffuse[channel] + gamma * vertexC.color_diffuse[channel];
          colorSpecularOfIntersection[channel] = alpha * vertexA.color_specular[channel] + beta * vertexB.color_specular[channel] + gamma * vertexC.color_specular[channel];
        }
        // printf("Intersection inside triangle! \n");
        return true;
      }  // intersection point P is inside the triangle. i.e. ray intersects with the triangle
    }


    // Find out the closest intersection point and its distance from ray source among ALL objects in the scene.
    // (there are only triangles and spheres primitives in the scene. )
    // Return true if intersecting with at least 1 object in the scene, false otherwise. 
    bool getClosestIntersection(glm::vec3 * closestIntersection, glm::vec3 * normalOfClosestIntersection,
                                double colorDiffuseOfClosestIntersection[3], double colorSpecularOfClosestIntersection[3], double & shininessOfClosestIntersection) {
      bool intersectionFlag = false;  // switch to on as long as there is 1 intersection
      *closestIntersection = glm::vec3(inf, inf, inf);  // Initialize closest intersection to be infinitely far. 
      float closestDistance = glm::length(*closestIntersection - origin);  // closest distance from origin of the ray to the intersection
      // must initialize. Otherwise dereferencing a null pointer inside the function will lead to seg fault. 
      glm::vec3 candidateIntersection(inf, inf, inf);  
      glm::vec3 candidateNormal(0.0, 0.0, 0.0);
      float candidateDistance;
      double candidateShininess; 
      double candidateColorDiffuse[3], candidateColorSpecular[3];

      // Loop through all spheres in the scene to find out the closest intersection of all spheres with this ray.
      for (int i = 0; i < num_spheres; i++) {
        // printf("Check intersection with sphere %d:\n", i);
        if (getSphereIntersection(spheres[i], &candidateIntersection, &candidateNormal)) {   // there is intersection
          intersectionFlag = true; 
          // Determine whether this intersection is < the running closest intersection
          candidateDistance = glm::length(candidateIntersection - origin);  
          if (candidateDistance < closestDistance) {  // update "closest"
            closestDistance = candidateDistance;
            *closestIntersection = candidateIntersection;
            *normalOfClosestIntersection = candidateNormal;
            // printf("Updating closest intersection = [%f, %f, %f], normal = [%f, %f, %f].\n", 
            //                (*closestIntersection).x, (*closestIntersection).y, (*closestIntersection).z, 
            //                (*normalOfClosestIntersection).x, (*normalOfClosestIntersection).y, (*normalOfClosestIntersection).z);
            for (int channel = 0; channel < 3; channel++) {
              colorDiffuseOfClosestIntersection[channel] = spheres[i].color_diffuse[channel];                
              colorSpecularOfClosestIntersection[channel] = spheres[i].color_specular[channel];
            }
            shininessOfClosestIntersection = spheres[i].shininess;
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
        if (getTriangleIntersection(triangles[i], &candidateIntersection, &candidateNormal, candidateColorDiffuse, candidateColorSpecular, candidateShininess)) {  // intersect
          intersectionFlag = true;
          // Determine whether this intersection is < the running closest intersection
          candidateDistance = glm::length(candidateIntersection - origin);  
          if (candidateDistance < closestDistance) {  // update "closest"
            closestDistance = candidateDistance;
            *closestIntersection = candidateIntersection;
            *normalOfClosestIntersection = candidateNormal;
            // printf("Updating closest intersection = [%f, %f, %f], normal = [%f, %f, %f].\n", 
            //                (*closestIntersection).x, (*closestIntersection).y, (*closestIntersection).z, 
            //                (*normalOfClosestIntersection).x, (*normalOfClosestIntersection).y, (*normalOfClosestIntersection).z);
            for (int channel = 0; channel < 3; channel++) {
              colorDiffuseOfClosestIntersection[channel] = candidateColorDiffuse[channel];                
              colorSpecularOfClosestIntersection[channel] = candidateColorSpecular[channel];
            }
            shininessOfClosestIntersection = candidateShininess;
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

// Helper function 
// When a light is NOT blocked, compute its contribution to the color of intersection point (from where the shadow ray is shoot) 
glm::vec3 applyPhongShading(Ray shadowRay, glm::vec3 lightPosition, glm::vec3 lightColor, glm::vec3 _cameraPosition,
                            glm::vec3 closestIntersection, glm::vec3 normalOfClosestIntersection, glm::vec3 viewVector,
                            double diffuseCoefficient[3], double specularCoefficient[3], double shininessCoefficient) {
                              
  glm::vec3 incomingLight = shadowRay.getPointPosition(1.0) - closestIntersection;  // then compute incoming light (NOTE it's a unit vector)
  
  // Compute the reflection light
  glm::vec3 reflectionLight = 2 * glm::dot(incomingLight, normalOfClosestIntersection) * normalOfClosestIntersection - incomingLight;
  
  // printf("Normal n = [%f, %f, %f], \n", normalOfClosestIntersection.x, normalOfClosestIntersection.y, normalOfClosestIntersection.z);
  // printf("Incoming light l = [%f, %f, %f], \n", incomingLight.x, incomingLight.y, incomingLight.z);
  // printf("Reflection light r = [%f, %f, %f], \n", reflectionLight.x, reflectionLight.y, reflectionLight.z);
  // printf("view vector v = [%f, %f, %f]. \n", viewVector.x, viewVector.y, viewVector.z);
  
  // Use Phong shading to determine the color of the pixel with respect to that light
  glm::vec3 phongShadingColor; 
  for (int channel = 0; channel < 3; channel++) {
    // printf("Light color channel %d value = %f, diffuse component = %f, specular component = %f \n", 
    //        channel, lightColor[channel], 
    //        static_cast<float>(diffuseCoefficient[channel]) * std::max(zero_threshold, glm::dot(incomingLight, normalOfClosestIntersection)), 
    //        static_cast<float>(specularCoefficient[channel]) * pow(std::max(zero_threshold, glm::dot(reflectionLight, viewVector)), shininessCoefficient));

    phongShadingColor[channel] = lightColor[channel] * (static_cast<float>(diffuseCoefficient[channel]) * std::max(zero_threshold, glm::dot(incomingLight, normalOfClosestIntersection)) 
                                              + static_cast<float>(specularCoefficient[channel]) * pow(std::max(zero_threshold, glm::dot(reflectionLight, viewVector)), shininessCoefficient));   
  }
  // printf("Phong shading color = [%f, %f, %f].\n", phongShadingColor.r, phongShadingColor.g, phongShadingColor.b);
  return phongShadingColor;
}




// Modularize camera ray casting given camera position and pixel position 
// return color obtained by this camera ray. 
glm::vec3 computeColorForCameraRay(glm::vec3 _cameraPosition, glm::vec3 _pixelPosition, int _reflectionDepth) {
  // Initiate pixel color as background color. 
  glm::vec3 color = background_color; 
  
  // 1) Cast a ray from camera position, whcih is (0, 0, 0) in camera frame the pixel center, 
  // printf("--------------------------\n");
  // printf("Step 1: Cast a ray from camera to pixel [%d, %d], which position is [%f, %f, %f] and check for intersection with objects in the scene.\n ", x, y, pixelCenterPos.x, pixelCenterPos.y, pixelCenterPos.z);
  Ray cameraRay = Ray(_cameraPosition, _pixelPosition - _cameraPosition);  
  // printf("Ray origin = [%f, %f, %f], direction = [%f, %f, %f]. \n", cameraRay.getOrigin().x, cameraRay.getOrigin().y, cameraRay.getOrigin().z, cameraRay.getDirection().x, cameraRay.getDirection().y, cameraRay.getDirection().z); 
  
  // Among ALL objects in the scene, find out the closest intersection of the backward ray shooting from the camera to the pixel. 
  glm::vec3 closestIntersection(inf, inf, inf);
  glm::vec3 normalOfClosestIntersection(0.0, 0.0, 0.0);
  double shininessCoefficient; 
  double diffuseCoefficient[3], specularCoefficient[3];
  if (cameraRay.getClosestIntersection(&closestIntersection, &normalOfClosestIntersection, diffuseCoefficient, specularCoefficient, shininessCoefficient)) {  
    // This is the case where backward ray hits some object in the scene, then its pixel color is determined by ALL the lights reflected by this object. 
    // we can "see" the object because of ambient light as a light source 
    color.r = static_cast<float>(ambient_light[0]); 
    color.g = static_cast<float>(ambient_light[1]);
    color.b = static_cast<float>(ambient_light[2]); 
    // printf("Backray hits an object! Start with ambient color = [%f, %f, %f].\n", color.r, color.g, color.b);

    // Once intersection is determined, view vector is the same regardless of light source
    glm::vec3 viewVector = glm::normalize(_cameraPosition - closestIntersection);  // recall view vector is from intersection point, towards camera eye

    // 2) Find out color contributed by all non-ambient light sources by shooting shadow ray.
    // printf("\nStep 2: Cast shadow ray for each light:\n");
    for (int i = 0; i < num_lights; i++) {  
      Light light = lights[i];
      glm::vec3 lightPosition = glm::vec3(static_cast<float>(light.position[0]), static_cast<float>(light.position[1]), static_cast<float>(light.position[2]));
      glm::vec3 lightColor = glm::vec3(static_cast<float>(light.color[0]), static_cast<float>(light.color[1]), static_cast<float>(light.color[2]));
      // cast a shadow ray from the closest intersection point to each light source
      Ray shadowRay = Ray(closestIntersection, lightPosition - closestIntersection);

      // Among ALL objects in the scene, find out the closest intersection of the shadow ray from intersection point to the light 
      glm::vec3 closestBlockageIntersection(inf, inf, inf);
      glm::vec3 normalOfClosestBlockageIntersection(0.0, 0.0, 0.0);
      double shininessCoefficientOfClosestBlockage; 
      double diffuseCoefficientOfClosestBlockage[3], specularCoefficientOfClosestBlockage[3];

      if (!shadowRay.getClosestIntersection(&closestBlockageIntersection, &normalOfClosestBlockageIntersection, diffuseCoefficientOfClosestBlockage, specularCoefficientOfClosestBlockage, shininessCoefficientOfClosestBlockage)) {
        // No objects blocking between intersection surface point and the light 
        // printf("Light %d (pos = [%f, %f, %f] is NOT blocked. Intersection point existing color = [%f, %f, %f]. \n", 
        //      i, lightPosition.x, lightPosition.y, lightPosition.z, lightColor.r, lightColor.g, lightColor.b);
        // Apply phong shading for the unblocked light onto the intersection point 
        glm::vec3 phongShadingColor = applyPhongShading(shadowRay, lightPosition, lightColor, _cameraPosition,
                          closestIntersection, normalOfClosestIntersection, viewVector,
                          diffuseCoefficient, specularCoefficient, shininessCoefficient);
        color = color + phongShadingColor;  // add to exisitng color of the intersection point                                            
    
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
          // Apply phong shading for the unblocked light onto the intersection point 
          glm::vec3 phongShadingColor = applyPhongShading(shadowRay, lightPosition, lightColor, _cameraPosition,
                                                          closestIntersection, normalOfClosestIntersection, viewVector,
                                                          diffuseCoefficient, specularCoefficient, shininessCoefficient);
          color = color + phongShadingColor;  // add to exisitng color of the intersection point
        } 
      }
      // printf("After shadow ray for light %d, color = [%f, %f, %f] \n",  i, color.r, color.g, color.b);
    } 

    // Extra credit: ray tracing (= ray casting + reflection + refraction). 
    if (_reflectionDepth < MAX_RECURSION) {
      // Cast a secondary ray by reflecting the view vector about the normal of the intersection point 
      glm::vec3 reflectViewVector = 2 * glm::dot(viewVector, normalOfClosestIntersection) * normalOfClosestIntersection - viewVector; 
      // The secondary ray is the new "camera ray", whose color will be added to the total color  
      color = glm::vec3(1.0 - reflection_color_weight, 1.0 - reflection_color_weight, 1.0 - reflection_color_weight) * color 
      + glm::vec3(reflection_color_weight, reflection_color_weight, reflection_color_weight) * computeColorForCameraRay(closestIntersection, closestIntersection + reflectViewVector, _reflectionDepth + 1);
    }
  } // backward ray does not hit any object in the scene, therefore pixel color is the background color (no modification to initial color)
  
  return color;
}

void draw_scene()
{ 
  // Calculate boundaries of camera frame
  float yMax = tan(fov / 2); // printf("yMax = %f\n", yMax);
  float xMax = yMax * aspect_ratio; // printf("xMax = %f\n", xMax);
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
      glm::vec3 color; 
      if (anti_aliasing) {  
        // Sample points inside the pixel box, and shoot camera ray for every sampled points on the image plane.  
        float xOffsets[anti_aliasing_sample_size], yOffsets[anti_aliasing_sample_size];  // x, y, are both [0, 1]
        getHaltonSequence2D(anti_aliasing_sample_size, xOffsets, yOffsets);
        for (int s = 0; s < anti_aliasing_sample_size; s++) {
          glm::vec3 samplePixelPos = pixelBottomLeftPos + glm::vec3(xOffsets[s] * pixelWidth, yOffsets[s] * pixelHeight, 0);
          glm::vec3 sampleColor = computeColorForCameraRay(camera_position, samplePixelPos, 0);
          color += sampleColor / (float) anti_aliasing_sample_size;  // weighted by sample size, and added to the pixel color 
        }
      } else {  // shoot 1 camera ray towards the center of pixel box 
        glm::vec3 pixelPos = pixelBottomLeftPos + glm::vec3(pixelWidth / 2, pixelHeight / 2, 0);   
        color = computeColorForCameraRay(camera_position, pixelPos, 0);
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
  if(strcasecmp(expected,found))
  {
    printf("Expected '%s ' found '%s '\n", expected, found);
    printf("Parsing error; abnormal program abortion.\n");
    exit(0);
  }
}

void parse_doubles(FILE* file, const char *check, double p[3])
{
  char str[100];
  fscanf(file,"%s",str);
  parse_check(check,str);
  fscanf(file,"%lf %lf %lf",&p[0],&p[1],&p[2]);
  printf("%s %lf %lf %lf\n",check,p[0],p[1],p[2]);
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

int loadScene(char *argv)
{
  FILE * file = fopen(argv,"r");
  if (!file)
  {
    printf("Unable to open input file %s. Program exiting.\n", argv);
    exit(0);
  }

  int number_of_objects;
  char type[50];
  Triangle t;
  Sphere s;
  Light l;
  fscanf(file,"%i", &number_of_objects);

  printf("number of objects: %i\n",number_of_objects);

  parse_doubles(file,"amb:",ambient_light);

  for(int i=0; i<number_of_objects; i++)
  {
    fscanf(file,"%s\n",type);
    printf("%s\n",type);
    if(strcasecmp(type,"triangle")==0)
    {
      printf("found triangle\n");
      for(int j=0;j < 3;j++)
      {
        parse_doubles(file,"pos:",t.v[j].position);
        parse_doubles(file,"nor:",t.v[j].normal);
        parse_doubles(file,"dif:",t.v[j].color_diffuse);
        parse_doubles(file,"spe:",t.v[j].color_specular);
        parse_shi(file,&t.v[j].shininess);
      }

      if(num_triangles == MAX_TRIANGLES)
      {
        printf("too many triangles, you should increase MAX_TRIANGLES!\n");
        exit(0);
      }
      triangles[num_triangles++] = t;
    }
    else if(strcasecmp(type,"sphere")==0)
    {
      printf("found sphere\n");

      parse_doubles(file,"pos:",s.position);
      parse_rad(file,&s.radius);
      parse_doubles(file,"dif:",s.color_diffuse);
      parse_doubles(file,"spe:",s.color_specular);
      parse_shi(file,&s.shininess);

      if(num_spheres == MAX_SPHERES)
      {
        printf("too many spheres, you should increase MAX_SPHERES!\n");
        exit(0);
      }
      spheres[num_spheres++] = s;
    }
    else if(strcasecmp(type,"light")==0)
    {
      printf("found light\n");
      parse_doubles(file,"pos:",l.position);
      parse_doubles(file,"col:",l.color);

      if(num_lights == MAX_LIGHTS)
      {
        printf("too many lights, you should increase MAX_LIGHTS!\n");
        exit(0);
      }
      lights[num_lights++] = l;
    }
    else
    {
      printf("unknown type in scene description:\n%s\n",type);
      exit(0);
    }
  }
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
  printf("========== Parsing arguments:==============\n");
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
        if (argValue == "false") {anti_aliasing = false; }  // by default is true.
      }

      // Number of reflection bounces
      if (argFlag == "reflection-bounces") {
        MAX_RECURSION = std::stoi(argValue);
      }
 
    }
  }
  printf("========== All arguments have been parsed.==============\n");  



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

