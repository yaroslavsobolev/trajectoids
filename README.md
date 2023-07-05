# Trajectoids
Given a path, computes the 3D shape ("trajectoid") that would follow this path when rolling down a slope

Online demo of this code is available in [a Google Colab Notebook](https://colab.research.google.com/drive/1XZ7Lf6pZu6nzEuqt_dUCHormeSbCCMlP)
: <a target="_blank" href="https://colab.research.google.com/drive/1XZ7Lf6pZu6nzEuqt_dUCHormeSbCCMlP">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

DOI of respective research article: *to be added*

DOI of this repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8116412.svg)](https://doi.org/10.5281/zenodo.8116412)

## Installation

Clone (checkout) this repository after installing these dependencies:

`Python 3.7+`
k
`trimesh`(the only hard dependency)

`matplotlib` (if plotting is needed)

`scikit-image` (for loading an image as input path)

`mayavi` (for viewing the mappings of the trajectory onto the rolling sphere in 3D)

Full requirements are:

```
matplotlib~=3.1.0
numpy~=1.21.6
scipy~=1.5.2
tqdm~=4.64.1
scikit-image~=0.15.0
trimesh~=3.7.0
plotly~=5.8.2
mayavi~=4.7.2
scikit-learn~=0.21.2
numba~=0.45.1
```

Installation should take less than 30 minutes on "normal" desktop computer. 
Code has been tested on Windows 10 Pro 64-bit, Python 3.7 and with package versions listed above.

## Running the code

All scripts are intended to run with "current working directory" set to repository root in the
Python interpreter.

### Computing the trajectoid shape from a path
Example usage with loading input path from an image:
```
from compute-trajectoids import *
input_path = get_trajectory_from_raster_image('vector_images/ibs_particle/ibs_v5_current_good.png')
compute_shape(input_path, 
              kx=1.0678, 
              ky=0.8009, 
              folder_for_path='trajectory_project_1',
              folder_for_meshes='cutting_boxes')
```

This will compute the same number of "cutting boxes" as the number of points in the `input_path` and save 
these boxes as `.obj` files into the `folder_for_meshes` directory. For obtaining the final trajetroid mesh,
these boxes must be subtracted from a sphere of radius `R` greater than the core radius `r=1`. For this putpose,
we imported all the boxes into Autodesk 3ds Max 2018 using the
["Batch Export/Import v04.12" plugin](https://www.scriptspot.com/3ds-max/scripts/batch-exportimport) and then 
used built-in boolean operators.

For reproducing the shape calculations of the trajectoids described in the paper, run respective `.py`  scripts
in the `examples` folder. Each script is for one trajectoid and uses data from (and then outputs into) 
the respective project directory in the `examples` subfolder. These scripts should be run with
"current working directory" set to repository root, not to the `examples` folder.

### Testing whether a two-period trajectoid exists for a various path types

See `existence_testing.py`. Uncomment the parts of `main` that check existence for the path type you are
interested in and run. This script also contains functions for making animations shown in the Movie 3 of the paper.
Typically takes less than 10 minutes on "normal" desktop computer, though execution time can be longer when testing a
large range of scales with small step.

<!---
## Citation
If you use this code, please cite our paper:
```
@article{2021trajectoids,
  title={Solid-body trajectoids shaped to roll along desired pathways: downwards, upwards, and in loops},
  author={Sobolev, Yaroslav I. and Dong, Ruoyu and Granick, Steve and Grzybowski, Bartosz A.},
  journal={XXX},
  year={2022}
}
```
-->

### Tracking the center of mass on experimental videos

See the `trajectory_analysis.py` script. To reproduce the tracking of specific experiments from the paper, uncompress
the `frames.mp4` in `example/EXAMPLE_FOLDER/video/frames` into separate `.jpg` frames, then
uncomment the respective lines at the end of `trajectory_analysis.py` script and run it. 
The raw frames are not included into this repository because of their size (about half a gigabyte per experiment)
but are available from Yaroslav (`yaroslav.sobolev@gmail.com`) on request.
Should take less than half an hour on a "normal" desktop computer for <1000 frames in experimental video.

## 3D printing

Before you attempt to print trajectoids, 
make sure that you are able to 3D print and assemble a sufficiently precise sphere of outer radius 15.875 mm 
having a concentric spherical cavity housing a 1-inch ball bearing (25.4 mm diameter). 
If you intend make trajectoids with a different diameter of ball bearing, or a different radius r, 
then use your intended values for this test sphere too.
If your test sphere does not roll satisfactorily down a 0.5-degree slope, your trajectoids will not work, either. 
If maximum angle `β_max` between gravity projection and your trajectoid's path directions is large -- close to 90 degrees -- then
your test sphere must perform good at even smaller slopes for your trajectoid to perform well. 
More specifically, if you intend to run your trajectoid on a slope having angle `α`, then your test sphere must be capable 
of performing at slope `γ = α * cos(β_max)`, assuming that α is small. 

If your sphere gets "stuck" at certain orientations instead of rolling down continuously, 
your trajectoids will have the same problem too.
Typically, there are two possible reasons of test sphere's poor performance:

* Your ball bearing is not positioned concentrically with the outer 3D printed surface.
* Your outer surface is not spherical -- probably it's an ellipsoid instead.

Make appropriate corrections to your 3D printer calibration (or to scales along X,Y,Z axes) until you succeed in
manufacturing a test sphere with satisfactory rolling performance.

