# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#

#####################################################################
def calibrate(im, XYZ, uv):
    # TBD
    assert(len(XYZ) == len(uv))
    XYZ_h = np.array([list(i + (1,)) for i in XYZ]) #Convert to homogeneous
    uv_h = np.array([u + (1,) for u in uv])
        
    #Initialize A, append all Ai
    A = []
    for i in range(len(XYZ)):
        world = XYZ_h[i][0:3]
        img = uv_h[i][0:2]
        
        (u,v) = img
        (X,Y,Z) = world
        
        Ai = np.array([0,0,0,0,-X,-Y,-Z,-1, v*X, v*Y, v*Z,v, X,Y,Z,1,0,0,0,0,-u*X, -u*Y,-u*Z,-u])
        A.append(Ai)
    
    A = np.array(A).T
    U,S,V = np.linalg.svd(A) #Calculate SVD
    P = V[-1] / np.linalg.norm(V[-1])
    p = np.reshape(P, (3,4))
    C = p
    projection = (C @ XYZ_h.T).T
    projection = [x[0:2] for x in projection]
    print(projection)
    
    from vgg_KR_from_P import vgg_KR_from_P
    K,R,t = vgg_KR_from_P(p)
    
    print("K: " + str(K))
    print("R: " + str(R))
    print("t: " + str(t))
    return C
'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% your name, date 
'''

############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    H = None
    return H 

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% your name, date 
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

#main loop
I = Image.open('stereo2012a.jpg');
RealCoords = [(7,7,0), (14,7,0), (21,7,0), (7,14,0), (14,14,0), (21,14,0), (0,7,7), (0,7,14), (0,7,21), (0,14,7), (0,14,14), (0,14,21)]
print("Please click on the first 2 rows of XY dots, then the first two rows of ZY dots")
import time
time.sleep(5)
plt.imshow(I)
uv = plt.ginput(12) # Graphical user interface to get 6 points
calibrate(I, RealCoords, uv)