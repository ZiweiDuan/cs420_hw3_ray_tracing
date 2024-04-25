Assignment #3: Ray tracing

FULL NAME: Ziwei Duan


MANDATORY FEATURES
------------------

<Under "Status" please indicate whether it has been implemented and is
functioning correctly.  If not, please explain the current status.>

Feature:                                 Status: finish? (yes/no)
-------------------------------------    -------------------------
1) Ray tracing triangles                  yes

2) Ray tracing sphere                     yes

3) Triangle Phong Shading                 yes

4) Sphere Phong Shading                   yes

5) Shadows rays                           yes

6) Still images                           yes

- For core credit, please refer to jpeg suffixed with _core_credit.jpeg.
   
7) Extra Credit (up to 30 points)
- Good antialiasing 
    - Image output: snow_anti_aliasing.jpeg, table_anti_aliasing.jpeg.
- Monte-Carlo sampling 
    - Please `cd ../hw3-starterCode-BRDF`, and run `./hw3_brdf test2.scene`
    - You can change the number of camera rays shoot per pixel by running `./hw3_brdf test2.scene test2_spp35.jpeg --brdf_sample_size=35`; 
    - You can see how test2 renderings progress as we increase the number of camera rays: 
      Image output include: test2_spp5.jpeg, test2_spp15.jpeg, test2_spp25.jpeg, test2_spp35.jpeg, test2_spp50.jpeg, test2_spp100.jpeg, snow_spp50.jpeg.
    - I compile BRDF rendering separately to prevent accidently breaking the core credit code. 



   
