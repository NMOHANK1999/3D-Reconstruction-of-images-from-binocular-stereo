'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

# load Library 
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
from helper import *
from util import *

import numpy as np
import matplotlib.pyplot as plt
import submission as sub

# load corresponding points and images
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')


pts1 = data['pts1']
pts2 = data['pts2']


N = pts1.shape[0]
M = 640

print(pts1.shape)
print(pts2.shape)

#find the Fundamental matrix
# 2.1
F8 = sub.eightpoint(pts1, pts2, M)
print(F8)

I1 = np.array(im1)  # First image
I2 = np.array(im2)  # Second image
F = np.array(F8) 

# displayEpipolarF(I1, I2, F)
# plt.close('all')

#Output: Save your matrix F, scale M to the file q2 1.npz.
#np.savez("../data/results/q2_1.npz", F = F, M = M)

#load K1 and K2 intrinsic matrixes
intrinsic_matrix = np.load('../data/intrinsics.npz')
K1 = intrinsic_matrix['K1']
K2 = intrinsic_matrix['K2']

#find E
E = sub.essentialMatrix(F, K1, K2)
print("E = ", E)



#Find four possible solutions of M2(3X4), combinations of T1, T2 and R1, R2 
M2 = camera2(E)


#Find their corresponding C1 = K1@M1
M1 = np.column_stack([np.eye(3), np.zeros(3).reshape(-1, 1)])
C1 = K1 @ M1

# define some variables
lowest_err = float('inf') 
C2_optimal = []
M2_optimal = []
w_best = []

#find the C2 with the lowest reprojection err
for i in range(M2.shape[2]):
    C2 = K2 @ M2[:, :, i]
    w, err = sub.triangulate(C1, pts1, C2, pts2)
    if np.min(w[:, -1]) >=  0 :
        lowest_err = err
        C2_optimal = C2[:]
        M2_optimal = M2[:, :, i]
        w_best = w[:]
        
#Save the correct M2, the corresponding C2, and 3D points P to q3 3.npz
np.savez('../data/results/q3_3.npz', M2=M2_optimal, C2=C2_optimal, w_best= w_best)

#checking the epipolarCorrespondence function
pointselected, pointspredicted = epipolarMatchGUI(I1, I2, F)
plt.close('all')

np.savez('../data/results/q4_1.npz', F = F, pointselected = pointselected, pointspredicted = pointspredicted )

print(pointselected, pointspredicted)
