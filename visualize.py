'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''


import matplotlib
#matplotlib.use('Qt5Agg')


import numpy as np
import matplotlib.pyplot as plt
import submission as sub
from helper import *
from util import *

data = np.load('../data/some_corresp.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

pts1 = data['pts1']
pts2 = data['pts2']

# 2.1
F8 = sub.eightpoint(pts1 ,pts2 , M)

I1 = np.array(im1)  # First image
I2 = np.array(im2)  # Second image
F = np.array(F8) 

displayEpipolarF(I1, I2, F)
plt.close('all')

#Output: Save your matrix F, scale M to the file q2 1.npz.
np.savez("../data/results/q2_1.npz", F = F, M = M)

intrinsic_matrix = np.load('../data/intrinsics.npz')
K1 = intrinsic_matrix['K1']
K2 = intrinsic_matrix['K2']

E = sub.essentialMatrix(F, K1, K2)

#Output: Save your estimated E using F from the eight-point algorithm to q3 1.npz
np.save("../data/results/q3_1.npz", E)
print(E)

M2_s = camera2(E)
print(M2_s)

#Find their corresponding C1 = K1@M1
M1 = np.column_stack([np.eye(3), np.zeros(3).reshape(-1, 1)])
C1 = K1 @ M1


# find the corresponding tcx1 and tcy1, tcx2 and tcy2

TempleCoords = np.load("../data/templeCoords.npz")
tcx1 = TempleCoords['x1'][:, 0]
tcy1 = TempleCoords['y1'][:, 0]
templecoordspoints_1 = np.column_stack([tcx1, tcy1])

tcx2 = []
tcy2 = []

for i in range(len(tcx1)):
    x2_pred, y2_pred = sub.epipolarCorrespondence(im1, im2, F, int(tcx1[i]), int(tcy1[i]))
    tcx2.append(x2_pred)
    tcy2.append(y2_pred)
    
tcx2 = np.array(tcx2)
tcy2 = np.array(tcy2)
templecoordspoints_2 = np.column_stack([tcx2, tcy2])

# find the best c2 for the points
# define some variables

# lowest_err_temple_coor = float('inf') 
# C2_optimal_temple_coor = []
# w_best_temple_coor = []

M2 = None
w_best_temple_coor = None

#find the C2 with the lowest reprojection err
for i in range(M2_s.shape[2]):
    M2_temp = M2_s[:, :, i]
    C2_temp = K2 @ M2_temp
    w, err = sub.triangulate(C1, templecoordspoints_1, C2_temp, templecoordspoints_2)
    
    if np.min(w[:, -1]) > 0: #initially this was the least reprojection error but, realized that just because something has the least error doesnt mean its correct, z min being greater that zero was a better analysis
        M2 = M2_temp
        w_best_temple_coor = w
        break
        
        # lowest_err_temple_coor = err
        # C2_optimal_temple_coor = C2[:]
        # w_best_temple_coor = w[:]
        
    
# #checking the epipolarCorrespondence function
# epipolarMatchGUI(I1, I2, F)
# plt.close('all')


C2 = np.dot(K2, M2)

np.savez('../data/results/q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)


#plotting the world points using scatter plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = w_best_temple_coor[:, 0]
y = w_best_temple_coor[:, 1]
z = w_best_temple_coor[:, 2]
ax.scatter(x, y, z, c='b', marker='.')

ax.set_title('3D reconstruction along X, Y, Z')

plt.show()
