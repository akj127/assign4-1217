Readme for pose estimation function
"""
    Estimate the pose of the camera based on feature correspondences between consecutive frames using a RANSAC-based approach.

    Parameters:
    - features_coor (numpy.ndarray): Array containing the coordinates of matched features in the previous and current frames. The shape of the array should be (N, 8), where N is the number of matched features. Each row represents the coordinates of a matched feature in the format [prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y].

    Returns:
    - C (numpy.ndarray): Estimated rotation matrix representing the transformation from the previous camera frame to the current camera frame.
    - r (numpy.ndarray): Estimated translation vector representing the translation from the previous camera frame to the current camera frame.
    - f_r_prev (numpy.ndarray): Array containing the coordinates of features in the previous right image used for the estimation of pose. Each row represents the (x, y) coordinates of a feature.
    - f_r_cur (numpy.ndarray): Array containing the coordinates of features in the current right image used for the estimation of pose. Each row represents the (x, y) coordinates of a feature.

    This function estimates the camera pose by finding a transformation that minimizes the reprojection error between matched features in consecutive frames. It employs a RANSAC-based approach to robustly estimate the pose in the presence of outliers.

    The function iterates over a fixed number of iterations, randomly selecting subsets of matched features to estimate a transformation. The transformation with the maximum number of inliers, i.e., matched features consistent with the estimated transformation, is selected as the final pose estimation.

    Within each iteration, the function computes the transformation matrix C and translation vector r using a closed-form solution derived from corresponding 3D points reconstructed from stereo image pairs. It then calculates the reprojection error for each matched feature and considers it an inlier if the error is below a certain threshold.

    The estimated rotation matrix C and translation vector r represent the transformation from the previous camera frame to the current camera frame. Additionally, the function returns arrays containing the coordinates of features used for pose estimation in both the previous and current right images.
    """
