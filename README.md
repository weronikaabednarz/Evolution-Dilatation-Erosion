# Evolution Dilatation Erosion

**Project subject: Contextual and morphological transformations.**

The projects included operations:
- dilatation and erosion together with a neighbourhood of radius 1 (there are 8 neighbours per pixel, the so-called Moore neighbourhood),
- opening and morphological closure,
- calculation of the convolution of a function with a mask of radius 1,
- calculation of the convolution of a function with a mask of radius r.

It also includes the possibility to change the neighbourhood radius for morphological operations, loading predefined masks/filters with radius = 1 or different radius from a file.

In addition, it takes into account operations for applying different filters on the bitmap:
- Gauss,
- low-pass
- high-pass.

Technologies used in the project: **Python** with libraries:
- numpy - a library for scientific calculations, operations on multidimensional arrays and matrices
- matplotlib - a module for creating graphs
- PIL - a module for image handling

Erosion picture:

![1](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/2_erozja.bmp)

Dilatation picture:

![2](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/1_dylatacja.bmp)

Dilatation of erosion picture:

![3](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/3_dylatacja_erozji.bmp)

Erosion of dilatation picture:

![4](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/4_erozja_dylatacji.bmp)

Weave function - gaussfilter:

![5](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/11_splot_dla_gaussfilter.bmp)

Weave function - upperfilter:

![6](https://github.com/weronikaabednarz/Evolution-Dilatation-Erosion/blob/main/14_splot_dla_upperfilter.bmp)
