Assignment #3: Ray tracing
Ziwei Duan

**CORE FEATURES:**
------------------
1) Ray tracing triangles                 

2) Ray tracing sphere                     

3) Triangle Phong Shading                

4) Sphere Phong Shading                  

5) Shadows rays                          

6) Still images                           

- For core credit, please refer to jpeg suffixed with _core_credit.jpeg.


**Extra Credit Features:** (up to 30 points)
- Good antialiasing
    - This feature is turned on by default. You can also explicitly turn on the feature by `./hw3_brdf test2.scene test2.jpeg --anti-aliasing=true`;
    - Image output: snow_anti_aliasing.jpeg, table_anti_aliasing.jpeg.
- Recursive Reflection:
   - `cd ./hw3-starterCode`, and you can specify the maximum number of reflection rays allowed:  `./hw3_brdf test2.scene test2.jpeg --reflection-bounces=2`. By default `--reflection-bounces=0`;     
- Monte-Carlo sampling 
    - Please `cd ../hw3-starterCode-BRDF`, and run `./hw3_brdf test2.scene`
    - You can change the number of camera rays shoot per pixel by running `./hw3_brdf test2.scene test2_spp35.jpeg --brdf_sample_size=35`; 
    - You can see how test2 renderings progress as we increase the number of camera rays: 
      Image output include: test2_spp5.jpeg, test2_spp15.jpeg, test2_spp25.jpeg, test2_spp35.jpeg, test2_spp50.jpeg, test2_spp100.jpeg, snow_spp50.jpeg, etc. 
    - I compile BRDF rendering separately to prevent accidently breaking the core credit code.




   
