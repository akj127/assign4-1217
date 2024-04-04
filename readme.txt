### README: Visual Odometry Functions

This README provides a brief overview of the key functions for visual odometry.

---

### `pose_estimation(features_coor)`

- Estimates the vehicle's pose (rotation and translation) using feature correspondences between consecutive frames.

### `calc_transform(prevCoor, currCoor, n)`

- Calculates the transformation between two sets of 3D coordinates using Singular Value Decomposition (SVD).

### `threeD_calc(fl, fr, n)`

- Computes the 3D coordinates of features from their pixel coordinates in left and right images.

---

These functions are essential for estimating the vehicle's motion and generating 3D information from image data.