# buddhabrot-opencl
Utitilty to generate density plot of mandelbrot set orbits on a GPU using OpenCL.

Original CPU based renderer: https://github.com/mallocc/buddhabrot

-----

|![](https://github.com/mallocc/buddhabrot/blob/main/tour_demo.gif)|
|:--:|
|`buddhabrot.exe -w 100 -h 100 -s 50000 -i 1000 -et 5 -o output/test1_ -steps 24 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 96 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 48 -nc -b 0 -steps 12 -nc -steps 96 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629` ~60s render time|
|![](https://github.com/mallocc/buddhabrot/blob/main/tour_demo_colour.gif)|
|`buddhabrot.exe -w 200 -h 200 -s 1000000 -ir 2000 -ig 200 -ib 20 -o output/test2_ -steps 24 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 96 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 48 -nc -b 0 -steps 12 -nc -steps 96 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629` ~35m render time|
|![](https://github.com/mallocc/buddhabrot/blob/main/nebula_demo.png)|
|`buddhabrot.exe -w 720 -h 720 -s 100000000 -ir 2000 -ig 200 -ib 20 -x0 -2 -y0 -1.5 -x1 1 -y1 1.5` ~10m render time|

-----

# Intro
This program aims to make it easy to generate buddhabrot images and compile animations. It has various options that can be changed to stylise your image. Transformations can be made to the render, such as translation and rotation. Animations can be made using the staging feature, where images are interpolated between key frames (stages).

# Dependancies
The only dependancy is the famous `stb_image_write.h` header, for creating the image files. Also, I've added my solution file for anyone that likes Visual Studio.

# Getting started
I've added a shell script `test.sh` that will attempt to generate the anination gif above, so to check everything is working, and to see an example usage.

# Usage/Options

We can add many options to the program to specify how we want the image to look:

 - `-w | --width` - the width of the image (px)
 - `-h | --height` - the height of the image (px)
 - `-i | --iterations` - max iteratons used for a greyscale image
 - `-ir | --iterations-red` - max iteratons used for the red channel on an RGB image
 - `-ig | --iterations-green` - max iteratons used for the green channel on an RGB image
 - `-ib | --iterations-blue` - max iteratons used for the blue channel on an RGB image
 - `-im | --iterations-min` - min iterations used
 - `--gamma` - gamma correction value for the colouring (used for 'sqrt colouring')
 - `--radius` - radius bounds used in the bailout condition
 - `-x0 | -re0 | --real0` - (STAGE) top LEFT corresponding complex coordinate value
 - `-y0 | -im0 | --imaginary0` - (STAGE) TOP left corresponding complex coordinate value
 - `-x1 | -re1 | --real1` - (STAGE) bottom RIGHT corresponding complex coordinate value
 - `-y1 | -im1 | --imaginary1` - (STAGE) BOTTOM right corresponding complex coordinate value
 - `-s | --samples` - samples used for each channel
 - `-o | --output` - output filename used (for animation, incremented integer will be a appended to the end, i.e. test0.png, test1.png, ...)
 - `--steps` - (STAGE) steps for the current stage in the animation
 - `-a | --alpha` - (STAGE) alpha rotation in degrees
 - `-b | --beta` - (STAGE) beta rotation in degrees
 - `-t | --theta` - (STAGE) theta rotation in degrees (offsets of alpha an beta equally)
 - `-et | --escape-trajectories` - for greyscale images, filters for iterations above this value (but below max iterations)
 - `-etr | --escape-trajectories-red` - for red channel, filters for iterations above this value (but below max red iterations)
 - `-etg | --escape-trajectories-green` - for green channel, filters for iterations above this value (but below max green iterations)
 - `-etb | --escape-trajectories-blue` - for blue channel, filters for iterations above this value (but below max blue iterations)
 - `--counter` - offsets the incremeted integer used when generating animation images (if you need to append to an existing set of images)
 - `-n | --next | --next-stage` - will push all of the STAGE options to the list, and reset values for the next stage in the options
 - `-nc | --next-cpy | --next-stage-copy` - will push all of the STAGE options to the list, but will keep the values from the last stage (useful for appending stages in the options)
 
# Algorithm

## Mandelbrot Set
Consider the standard Mandelbrot algorithm:
```c++
for (int x = 0; x < w; ++x)
{
  for (int y = 0; y < h; ++)
  {
    z_re = c_re = (x / w) * (x1 - x0) + x0;
    z_im = c_im = (y / h) * (y1 - y0) + y0;

    int i = 1;
    for (; i < maxIterations && (z_re * z_re + z_im * z_im) < (radius * radius); ++i)
    {
      z_re = z_re * zre - z_im * z_im + c_re;
      z_im = 2 * z_re * z_im + c_im;
    }
    
    if (i < maxIteration)
      pixelData[x + y * w] = i % MAX_PIXEL_VALUE; // apply your colouring technique here
  }
}
```
which generates the usual Mandelbrot set we are familiar with. It will iterate over each pixel on the screen, and calculate the pixel colour based on how many times it iterated.

The idea is to work out where the corresponding pixel position in the complex plane ends up, and then colour the pixel depending on how far it went. Now there are two ways in which the trajectory stops:
 1. trajectory diverges and ends up traveling outside the radius bounds,
 2. or trajectory converges and ends up being attacted to a single point and can take potentially infinite iterations.
 
Usually colouring happens when 1. occurs for the pixel, and 2. means it hit infinity and should be coloured black. Bit like a black hole if it were a gravity field.

## Buddhabrot (Density Plot of Mandelbrot Set)

Theres is an interesting way to visualise the Mandelbrot set, which is to plot the density of each trajectory as they pass over the pixels. This is a sort of heat map of where the trajectories fall. This can be easily achieved by modifing the Mandelbrot algorithm to store the trajectory points:
```c++
float * t_re = new float[maxInterations];
float * t_im = new float[maxInterations];
for (int s = 0; s < samples; ++s)
{ 
    z_re = c_re = randf() * (x1 - x0) + x0;
    z_im = c_im = randf() * (y1 - y0) + y0;

    int i = 0;
    for (; i < maxIterations && (z_re * z_re + z_im * z_im) < (radius * radius); ++i)
    {
      t_re[i] = z_re = z_re * zre - z_im * z_im + c_re;
      t_im[i] = z_im = 2 * z_re * z_im + c_im;
    }
    
    if (i < maxIteration)
      for (int j = 0; j < i; ++i)
      {
        int x = (t_re[j] - x0) / (w / (x1 - x0));
        int y = (t_im[j] - y0) / (h / (y1 - y0));
        density[x + y * w]++;
      }
}
```
Notice that we are now not interating over the pixel positions, but randomly picking a starting position. We do this because we are trying to create a probability distribution of where trajectories fall. Using random samples helps us with this, because as the sample amount tends to infinity, the noise of the distribution falls to zero (like with any other distrubution).
The only thing that makes it slightly more complicated is the conversion of the complex position back to screen space, but it's just a case rearranging the screen-to-complex forumla used in the Mandelbrot code.

|![](https://github.com/mallocc/buddhabrot/blob/main/high_iterations.png)|
|:-:|
| Large amount of iterations used (100000) shows stable trajectories generating a lot of 'heat'. |

## Colouring
The sqrt colouring method is commonly used with buddhabrot data. It is as simple as finding the highest value that occurs in the buddhabrot data, and then for each apply: `pixel[x + y * w] = sqrt(density[x + y * w]) / sqrt(maxDensity) * MAX_PIXEL_VALUE;`. Gamma correction can be applied using: `pixel[x + y * w] = pow(density[x + y * w] / maxDensity, 1 / gamma) * MAX_PIXEL_VALUE;`.

|![](https://github.com/mallocc/buddhabrot/blob/main/gamma.gif)|
|:-:|
| Gamma value from 0 to 5. |

The examples talked about are for greyscale, but can easily scaled to use 3 colour components.

There is a very nice way to colour the data that makes it look like a space nebula. This is done by producing 3 images of the same position, but at varying iterations for each colour component. The example image at the start of the readme uses the RGB iteration values (2000, 200, 20). Finally, combine all the colour channels together in the same image.

## Rotation
Effectively, the buddhabrot can be treated as a 4d object, and be rotated in such, to produce unintuitive and complex transformations. 

Two of the basic intuitive axes of rotation (pitch and yaw) can be applied to the complex-to-screen coordinate conversion forumla:
```c++
int x = (t_re[j] * cos(alpha) + c_im * sin(alpha) - x0) / (w / (x1 - x0));
int y = (t_im[j] * cos(beta)  - c_re * sin(beta)  - y0) / (h / (y1 - y0));
```
where `c_` is the starting point for the trajectory.

We can rotate around a particular point of interest as well. This can be done by as easily as translating the trajectory point back to the origin, rotate and translate back again. The formula can be updated:
```c++
int x = ((t_re[j] - p_re) * cos(alpha) + (c_im - p_im) * sin(alpha) - x0) / (w / (x1 - x0)) + p_re;
int y = ((t_im[j] - p_im) * cos(beta)  - (c_re - p_re) * sin(beta)  - y0) / (h / (y1 - y0)) + p_im;
```
where `p_` is the point of interstion we want to rotate around. Note this is more readable in my source code!

|![](https://github.com/mallocc/buddhabrot/blob/main/rotation_cr.gif)|![](https://github.com/mallocc/buddhabrot/blob/main/rotation_ci.gif)|![](https://github.com/mallocc/buddhabrot/blob/main/rotation_mandelbrot.gif)|
|:-:|:-:|:-:|
| Rotation about the axis of `c_re`, from 0 to 180 degrees. | Rotation about the axis of `c_im`, from 0 to 180 degrees. | Rotation on both axes of 90 degrees results in the Mandelbrot Set. |


 
