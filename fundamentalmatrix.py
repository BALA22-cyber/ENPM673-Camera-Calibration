import cv2 as cv
import numpy as np
import random

def calculate_F_matrix(list_kp1, list_kp2):
        ###### Calculates the F matrix from a set of 8 points using SVD.
        ###### The rank is reduced from 3 to 2 for converging the epilines.

    A = np.zeros(shape=(len(list_kp1), 9))

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, S, VT = np.linalg.svd(A)
    F = VT[-1,:]
    F = F.reshape(3,3)
   
                            # Reducing rank from 3 to 2
    u, s, vt = np.linalg.svd(F)
    s[-1] = 0
    S = np.zeros((3,3))
    for i in range(3):
        S[i][i] = s[i]

    F = np.dot(u, np.dot(S, vt))
    return F

def RANSAC_F_matrix(x_a,x_b):
    S_in = []
    points = 8
    n = 0
    epsilon = .005
    max_pts = 1000
    iterations = 20
    Best_F = None

    for i in range(iterations):
        x1,x2 = [],[]
        mask = []

        for j in range(points):
            k = random.randint(0,len(x_a)-1)
            x1.append(x_a[k])
            x2.append(x_b[k])

        F = calculate_F_matrix(x1,x2)

        S = []
        for p1,p2 in zip(x_a,x_b):
            x1j = np.array([p1[0],p1[1],1])
            x2j = np.array([p2[0],p2[1],1])
            val = abs(x2j.T @ F @ x1j)
            if val < epsilon:
                S.append([p1,p2])
                mask.append(1)
            else:
                mask.append(0)

        if len(S) > n:
            n = len(S)
            S_in = S
            Best_F = F
            best_mask = mask

        # if len(S) > max_pts:
        #     break

    return Best_F 

def ransac_f_matrix(pts1,pts2): 
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.RANSAC)
    return F, mask

def getEssentialMatrix(K1, K2, F):
    E_ = K2.T.dot(F).dot(K1)
    U,S,V = np.linalg.svd(E_)
    S = [1,1,0]
                        #corrected E matrix 
    E = np.dot(U,np.dot(np.diag(S),V))
    return E