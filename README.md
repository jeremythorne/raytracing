# Ray Tracing in Rust

This began as an implementation of the exercises in [Ray Tracing in One Weekend](
https://raytracing.github.io/books/RayTracingInOneWeekend.html) in Rust, but I've added my own extensions.

## STL models, cubes and K-d Trees
reading STL binary models, making cubes from triangles, using K-d Trees to accelerate ray-triangle interections within a mesh.

![Two cubes and a mug](cubes_and_mug.png)

## Ray Tracing in One Weekend renders 
![Chapter 13 final render - many spheres](chapter13.png)
![Chapter 11 final render - hollow glass, lambertian, and metal spheres with depth of field](chapter11.png)

Non exhaustive list of other RTIOW implementations:

* https://github.com/fralken/ray-tracing-in-one-weekend
* https://github.com/perliedman/raytracing-in-one-weekend
* https://github.com/alexislozano/raytracing
* https://bitshifter.github.io/2018/04/29/rust-ray-tracer-in-one-weekend/
* https://misterdanb.github.io/raytracinginrust/