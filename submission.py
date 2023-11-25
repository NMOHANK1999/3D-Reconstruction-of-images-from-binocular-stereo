"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here

import numpy as np
import matplotlib.pyplot as plt
from util import refineF, _singularize
import submission as sub

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    '''
    
    '''
    #normalizing points
    pts1_scaled = pts1 / M
    pts2_scaled = pts2 / M    

    #making the X and Y's
    x1 = np.array(pts1_scaled[:, 0])
    y1 = np.array(pts1_scaled[:, 1])
    x2 = np.array(pts2_scaled[:, 0])
    y2 = np.array(pts2_scaled[:, 1])

    #defining the A mattrix
    #A = np.column_stack([x2 * x1 , x2 * y1 , x2 , y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)])

    A = np.column_stack([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, np.ones_like(x1)])

    #SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    #enforce singularity
    #U, S, VT = np.linalg.svd(F)
    #S[-1] = 0
    #F_enf_sing = U @ np.diag(S) @ VT 
    
    F_enf_sing = _singularize(F)
    
    #Unnormalizing the F mat
    T = np.row_stack([[1/M, 0, 0],[ 0, 1/M, 0],[ 0, 0, 1]])
    F_unnorm = T.T @ F_enf_sing @ T

    #refinement
    F_refined = refineF(F_unnorm, pts1, pts2)
    
    return F_refined

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x1 , y1 = pts1[:, 0], pts1[:, 1]
    x2 , y2 = pts2[:, 0], pts2[:, 1]
    
    # Define the rows of A
    A1 = np.row_stack([C1[0, 0]-C1[2, 0]*x1, C1[0, 1]-C1[2, 1]*x1, C1[0, 2]-C1[2, 2]*x1, C1[0, 3]-C1[2, 3]*x1]).T
    A2 = np.row_stack([C1[1, 0]-C1[2, 0]*y1, C1[1, 1]-C1[2, 1]*y1, C1[1, 2]-C1[2, 2]*y1, C1[1, 3]-C1[2, 3]*y1]).T
    A3 = np.row_stack([C2[0, 0]-C2[2, 0]*x2, C2[0, 1]-C2[2, 1]*x2, C2[0, 2]-C2[2, 2]*x2, C2[0, 3]-C2[2, 3]*x2]).T
    A4 = np.row_stack([C2[1, 0]-C2[2, 0]*y2, C2[1, 1]-C2[2, 1]*y2, C2[1, 2]-C2[2, 2]*y2, C2[1, 3]-C2[2, 3]*y2]).T
    
    #find the 3d projection
    N = len(pts1)
    w = np.zeros([N, 3])
    for i in range(N):
        A = np.row_stack([A1[i, :], A2[i, :], A3[i, :], A4[i, :]])
        _, _, VT = np.linalg.svd(A)
        hom_proj = VT[-1, :]
        w[i, :] = hom_proj[:3]/hom_proj[-1]
        
    #projecting back and checking for error
    W = np.column_stack([w, np.ones(N).reshape(-1, 1)])
    err = 0
    for i in range(N):
        hom_proj1 = C1 @ W[i, :].T
        hom_proj2 = C2 @ W[i, :].T
        proj1 = (hom_proj1[:2]/hom_proj1[-1]).T
        proj2 = (hom_proj2[:2]/hom_proj2[-1]).T
        err += np.linalg.norm([(proj1 - pts1[i]), (proj2 - pts2[i])])
        
    return w, err
        
    


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    
    
    #define a y range to search for the patch above and below, i did this because there were some places that gave faulty results  
    Y_searchrange = 50
    
    # Make the guassian window:
    size = 5  
    std_dev = 1.0
    center = size // 2
    # Initialize an empty 2D array to store Gaussian values
    guass_window = np.empty((size, size))
    # Calculate Gaussian values and populate the matrix
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            gaussian_value = np.exp(-0.5 * (x**2 + y**2) / (std_dev**2)) / (2 * np.pi * std_dev**2)
            guass_window[i, j] = gaussian_value
    # Normalize the matrix so that it sums up to 1
    guass_window /= np.sum(guass_window)
    
    #reshape the guass array
    guass_window_reshaped = np.repeat(guass_window[: , : , np.newaxis], 3, axis = 2)
    
    #height and width of image 1 and 2
    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    
    #find the epipolar line
    points1 = np.row_stack([x1, y1, 1])
    line2 = F @ points1
    a, b, c = line2[0], line2[1], line2[2]
    
    #find the image 1 patch
    patch1 = im1[y1-center:y1+center+1, x1-center:x1+center+1]
    
    #defining Y and X search points, for x= (-by - c)/a
    #Y = np.array(range(y1-search_range, y1+search_range))
    Y = np.array(range(size , h2 - size))
    X = np.round(-(b * Y + c) / a)
    
    valid = (X >= center) & (X < w2 - center) & (Y >= center) & (Y < h2 - center)
    
    X, Y = X[valid].astype(int), Y[valid]
    
    # finding the patch2 corresponding to patch 1 by pixel subtraction
    err_lowest = float('inf')
    x2_pred, y2_pred = None, None 
    for i in range(len(X)):
        patch2 = im2[Y[i] - center: Y[i] + center + 1, X[i] - center: X[i] + center + 1, :]
        err = np.sum((patch1 - patch2)**2 * guass_window_reshaped)
        if err < err_lowest:
            x2_pred = X[i]
            y2_pred = Y[i]
            err_lowest = err
            
    return x2_pred, y2_pred
        
    
    
    
    
    
'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
