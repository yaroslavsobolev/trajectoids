## 3dsmax workflow ##

Object sizes:

    ghost sphere radius = 1.3
    inner sphere (for ball bearing) radius = 0.8
    elliptic core correction to compensate for 3D printer artefacts: majir axis over minor axis ratio is 1675/1606

Batch Export/Import macros, v04.12, parameters:

    ENABLED:
        - rename fo file name
    DISABLED:
        - everything else

OBJ import options:

    EBABLED:
        - Flip ZY axis
    DISABLED:
        - everything else

## 3d printing ##

Ultimaker Cura slicing settings:

    X scale: 1617%
    Y scale: 1602%
    Z scale: 1606%

## Experiments ##

Slope angle:

    0.59 degrees

Camera recording:

    Frame rate: 120 fps
    Exposure: 1/125
    Aperture: F/10
    iso: 1600
    Focal length: 105 mm

## Analysis ##

Errors of visible shape centroid vs. full 6D pose tracking:

    RMS error of centroids: 0.8449805557450757 mm
    RMS error of full tracking: 0.7647014634838443 mm