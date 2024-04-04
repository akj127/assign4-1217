"""
2021-02 -- Wenda Zhao, Miller Tang

This is the class for a steoro visual odometry designed 
for the course AER 1217H, Development of Autonomous UAS
https://carre.utoronto.ca/aer1217
"""
import numpy as np
import cv2 as cv
import sys
import random

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

np.random.rand(1217)

class StereoCamera:
    def __init__(self, baseline, focalLength, fx, fy, cu, cv):
        self.baseline = baseline
        self.f_len = focalLength
        self.fx = fx
        self.fy = fy
        self.cu = cu
        self.cv = cv

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame_left = None
        self.last_frame_left = None
        self.new_frame_right = None
        self.last_frame_right = None
        self.C = np.eye(3)                               # current rotation    (initiated to be eye matrix)
        self.r = np.zeros((3,1))                         # current translation (initiated to be zeros)
        self.kp_l_prev  = None                           # previous key points (left)
        self.des_l_prev = None                           # previous descriptor for key points (left)
        self.kp_r_prev  = None                           # previous key points (right)
        self.des_r_prev = None                           # previoud descriptor key points (right)
        self.detector = cv.xfeatures2d.SIFT_create()     # using sift for detection
        self.feature_color = (255, 191, 0)
        self.inlier_color = (32,165,218)

            
    def feature_detection(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        feature_image = cv.drawKeypoints(img,kp,None)
        return kp, des, feature_image

    def featureTracking(self, prev_kp, cur_kp, img, color=(0,255,0), alpha=0.5):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cover = np.zeros_like(img)
        # Draw the feature tracking 
        for i, (new, old) in enumerate(zip(cur_kp, prev_kp)):
            a, b = new.ravel()
            c, d = old.ravel()  
            a,b,c,d = int(a), int(b), int(c), int(d)
            cover = cv.line(cover, (a,b), (c,d), color, 2)
            cover = cv.circle(cover, (a,b), 3, color, -1)
        frame = cv.addWeighted(cover, alpha, img, 0.75, 0)
        
        return frame
    
    def find_feature_correspondences(self, kp_l_prev, des_l_prev, kp_r_prev, des_r_prev, kp_l, des_l, kp_r, des_r):
        VERTICAL_PX_BUFFER = 1                                # buffer for the epipolor constraint in number of pixels
        FAR_THRESH = 7                                        # 7 pixels is approximately 55m away from the camera 
        CLOSE_THRESH = 65                                     # 65 pixels is approximately 4.2m away from the camera
        
        nfeatures = len(kp_l)
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)        # BFMatcher for SIFT or SURF features matching

        ## using the current left image as the anchor image
        match_l_r = bf.match(des_l, des_r)                    # current left to current right
        match_l_l_prev = bf.match(des_l, des_l_prev)          # cur left to prev. left
        match_l_r_prev = bf.match(des_l, des_r_prev)          # cur left to prev. right

        kp_query_idx_l_r = [mat.queryIdx for mat in match_l_r]
        kp_query_idx_l_l_prev = [mat.queryIdx for mat in match_l_l_prev]
        kp_query_idx_l_r_prev = [mat.queryIdx for mat in match_l_r_prev]

        kp_train_idx_l_r = [mat.trainIdx for mat in match_l_r]
        kp_train_idx_l_l_prev = [mat.trainIdx for mat in match_l_l_prev]
        kp_train_idx_l_r_prev = [mat.trainIdx for mat in match_l_r_prev]

        ## loop through all the matched features to find common features
        features_coor = np.zeros((1,8))
        for pt_idx in np.arange(nfeatures):
            if (pt_idx in set(kp_query_idx_l_r)) and (pt_idx in set(kp_query_idx_l_l_prev)) and (pt_idx in set(kp_query_idx_l_r_prev)):
                temp_feature = np.zeros((1,8))
                temp_feature[:, 0:2] = kp_l_prev[kp_train_idx_l_l_prev[kp_query_idx_l_l_prev.index(pt_idx)]].pt 
                temp_feature[:, 2:4] = kp_r_prev[kp_train_idx_l_r_prev[kp_query_idx_l_r_prev.index(pt_idx)]].pt 
                temp_feature[:, 4:6] = kp_l[pt_idx].pt 
                temp_feature[:, 6:8] = kp_r[kp_train_idx_l_r[kp_query_idx_l_r.index(pt_idx)]].pt 
                features_coor = np.vstack((features_coor, temp_feature))
        features_coor = np.delete(features_coor, (0), axis=0)

        ##  additional filter to refine the feature coorespondences
        # 1. drop those features do NOT follow the epipolar constraint
        features_coor = features_coor[
                    (np.absolute(features_coor[:,1] - features_coor[:,3]) < VERTICAL_PX_BUFFER) &
                    (np.absolute(features_coor[:,5] - features_coor[:,7]) < VERTICAL_PX_BUFFER)]

        # 2. drop those features that are either too close or too far from the cameras
        features_coor = features_coor[
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,0] - features_coor[:,2]) < CLOSE_THRESH)]

        features_coor = features_coor[
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) > FAR_THRESH) & 
                    (np.absolute(features_coor[:,4] - features_coor[:,6]) < CLOSE_THRESH)]
        # features_coor:
        #   prev_l_x, prev_l_y, prev_r_x, prev_r_y, cur_l_x, cur_l_y, cur_r_x, cur_r_y
        return features_coor
    
    def select_random_rows(self,matrix, n):
        num_rows = matrix.shape[0]
        selected_indices = np.random.choice(num_rows, size=n, replace=False)
        selected_rows = matrix[selected_indices, :]
        return selected_rows
    
    def threeD_calc(self, fl,fr, n) :
        Out = np.eye(n,3) # Initialize the output array with dimensions n x 3
        for i in range(len(fl)) :
        # Compute the 3D coordinate of the feature using triangulation formula
            Ai =  np.dot((self.cam.baseline/(fl[i,0]-fr[i,0])) ,np.array(
            [
                0.5*(fl[i,0] + fr[i,0]) - self.cam.cu,
                (self.cam.fx/self.cam.fy)*(0.5*(fl[i,1] + fr[i,1])) - self.cam.cv,
                self.cam.fx
            ]
            ))
            Out[i,:] = Ai # Store the computed 3D coordinate in the output array
        return Out

    def calc_transform(self, prevCoor, currCoor, n) :
        # Calculate the mean of previous and current coordinates
        p_a = np.mean(prevCoor, axis=0)
        p_b = np.mean(currCoor, axis=0)

        # Compute the matrix W_ba
        sum_ba = 0
        for j in range(n):
            sum_ba += np.matmul((currCoor[j, :] - p_b).reshape(3,1), (prevCoor[j, :] - p_a).reshape(1,3))
        w_ba = sum_ba/n

        # Perform Singular Value Decomposition (SVD) on W_ba
        V, _, UT = np.linalg.svd(w_ba)

        # Calculate determinants of V and UT
        detV = np.linalg.det(V)
        detU = np.linalg.det(UT.T)
        # Construct matrix A with the required determinant
        MatA = np.eye(3,3)
        MatA[2,2] = detU*detV
        # Calculate the rotation matrix C_ba
        C_ba = np.matmul(np.matmul(V, MatA), UT)
        # Calculate the translation vector r_ba_a
        r_ba_a = -np.matmul(C_ba.T, p_b) + p_a

        return C_ba, r_ba_a


    def pose_estimation(self, features_coor):
        # dummy C and r
        C = np.eye(3)
        r = np.array([0,0,0])
        # ------------- start your code here -------------- #

        len_features = len(features_coor)

        inliers = 0
        prev_in = []
        curr_in = []

        for i in range(50):
            # Extract features for previous and current frames
            f_r_prev, f_r_cur = features_coor[:,2:4], features_coor[:,6:8]
            f_l_prev, f_l_cur = features_coor[:, 0:2], features_coor[:, 4:6]
            
            prevCoor = self.threeD_calc(f_l_prev, f_r_prev, len_features)
            currCoor = self.threeD_calc(f_l_cur, f_r_cur, len_features)
            
            random_indices = np.random.choice(len_features, size=3, replace=False)

            # Select random rows from prevCoor and currCoor
            random_prevCoor = prevCoor[random_indices, :]
            random_currCoor = currCoor[random_indices, :]
            
            #calculate transformation matrix
            C_ba, r_ba_a = self.calc_transform(random_prevCoor, random_currCoor, 3)

            inliers_temp= 0 
            prev_in_temp= []
            curr_in_temp= []
            fr_prev_temp= []
            fr_curr_temp= []
            
            for j in range(len_features):
                # Calculate error for each feature
                Err = np.matmul((currCoor[j, :] - (C_ba @ (prevCoor[j, :] - r_ba_a))).T, (currCoor[j, :] - (C_ba @ (prevCoor[j, :] - r_ba_a))))
                
                if Err < 0.1: # Threshold for inliers
                    prev_in_temp.append(prevCoor[j, :])
                    curr_in_temp.append(currCoor[j, :])
                    fr_prev_temp.append(f_r_prev[j, :])
                    fr_curr_temp.append(f_r_cur[j, :])
                    inliers_temp+= 1

            if inliers_temp > inliers:
                inliers = inliers_temp
                prev_in = prev_in_temp
                curr_in = curr_in_temp
        # Calculate final transformation using inliers
        C, rba_a = self.calc_transform(np.array(prev_in), np.array(curr_in), inliers)

        # Calculate translation vector
        r = - 1 * np.dot(C, rba_a)
        # replace (1) the dummy C and r to the estimated C and r. 
        #         (2) the original features to the filtered features
        return C, r, f_r_prev, f_r_cur
    
    def processFirstFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        
        self.frame_stage = STAGE_SECOND_FRAME
        return img_left, img_right
    
    def processSecondFrame(self, img_left, img_right):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)
        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
    
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6],img_left, color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right, color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r
        self.frame_stage = STAGE_DEFAULT_FRAME
        
        return img_l_tracking, img_r_tracking

    def processFrame(self, img_left, img_right, frame_id):
        kp_l, des_l, feature_l_img = self.feature_detection(img_left)

        kp_r, des_r, feature_r_img = self.feature_detection(img_right)
        
        # compute feature correspondance
        features_coor = self.find_feature_correspondences(self.kp_l_prev, self.des_l_prev,
                                                     self.kp_r_prev, self.des_r_prev,
                                                     kp_l, des_l, kp_r, des_r)
        # draw the feature tracking on the left img
        img_l_tracking = self.featureTracking(features_coor[:,0:2], features_coor[:,4:6], img_left,  color = self.feature_color)
        
        # lab4 assignment: compute the vehicle pose  
        [self.C, self.r, f_r_prev, f_r_cur] = self.pose_estimation(features_coor)
        
        # draw the feature (inliers) tracking on the right img
        img_r_tracking = self.featureTracking(f_r_prev, f_r_cur, img_right,  color = self.inlier_color, alpha=1.0)
        
        # update the key point features on both images
        self.kp_l_prev = kp_l
        self.des_l_prev = des_l
        self.kp_r_prev = kp_r
        self.des_r_prev = des_r

        return img_l_tracking, img_r_tracking
    
    def update(self, img_left, img_right, frame_id):
               
        self.new_frame_left = img_left
        self.new_frame_right = img_right
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            frame_left, frame_right = self.processFrame(img_left, img_right, frame_id)
            
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            frame_left, frame_right = self.processSecondFrame(img_left, img_right)
            
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            frame_left, frame_right = self.processFirstFrame(img_left, img_right)
            
        self.last_frame_left = self.new_frame_left
        self.last_frame_right= self.new_frame_right
        
        return frame_left, frame_right 


