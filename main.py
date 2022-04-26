import cv2 as cv
import random
import numpy as np
import matplotlib.pyplot as plt
from fundamentalmatrix import *
from disparity import *
from Disparitycorrespondance import *
import time
from depth import *

def rescale(frame, scale):
    width = int (frame.shape[1]*scale)
    height = int (frame.shape[0]*scale)
    dimensions = (width , height)

    return cv.resize (frame, dimensions, interpolation = cv.INTER_AREA)

# def siftFeatures2Array(sift_matches, kp1, kp2):
#     matched_pairs = []
#     for i, m1 in enumerate(sift_matches):
#         pt1 = kp1[m1.queryIdx].pt
#         pt2 = kp2[m1.trainIdx].pt
#         matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
#     matched_pairs = np.array(matched_pairs).reshape(-1, 4)
#     return matched_pairs

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def extractCameraPose(E):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    U,D,Vt = np.linalg.svd(E)

    C_a = U[:,2].reshape(3,1)
    C_b = -U[:,2].reshape(3,1)

    R_a = U @ W @ Vt
    R_b = U @ W.T @ Vt

    Cset = [C_a,C_b,C_a,C_b]
    Rset = [R_a,R_a,R_b,R_b]

    if int(np.linalg.det(R_a)) == -1:
        for i in range(2):
            Cset[i] = -Cset[i]
            Rset[i] = -Rset[i]

    if int(np.linalg.det(R_b)) == -1:
        for i in range(2,4):
            Cset[i] = -Cset[i]
            Rset[i] = -Rset[i]


    return Cset,Rset


# print("Enter dataset no: ")
no = int(input('select data set\n1)curule\n2)octagon\n3)pendulum\n: '))
    # no = 1

if no == 1:
    # folder = './curule'
    K1 = np.array([[1758.23, 0, 977.42], 
                    [0, 1758.23, 552.15], 
                    [0, 0 ,1]])
    K2 = K1
    imgA = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/curule/im0.png",0)
    imgB = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/curule/im1.png",0)
    baseline = 88.39

elif no == 2:
    
    # folder = './octagon'
    K1 = np.array([[1742.11, 0, 804.90], 
                    [0, 1742.11, 541.22], 
                    [0, 0 ,1]])
    K2 = K1
    imgA = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/octagon/im0.png",0)
    imgB = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/octagon/im1.png",0)
    baseline = 221.76

elif no == 3:
    # folder = './pendulum'
    K1 = np.array([[1729.05, 0, -364.24], 
                    [0, 1729.05, 552.22], 
                    [0, 0 ,1]])
    K2 = K1
    imgA = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/pendulum/im0.png",0)
    imgB = cv.imread("/home/kb2205/Desktop/ENPM 673/Project3/ENPM673 pro3 dataset/pendulum/im1.png",0)
    baseline = 537.75

else:
    print('wrong input')
    exit()

f = K1[0,0]

# ### Feature extraction
imA = rescale(imgA,0.5)
imB = rescale(imgB,0.5)

sift = cv.SIFT_create()
                            # find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imA,None)
kp2, des2 = sift.detectAndCompute(imB,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)


####  Feature Matching
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2,k =2)

pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = ransac_f_matrix(pts1,pts2)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

ratio_threshold = 0.3
filtered_matches = []
for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        filtered_matches.append(m)

print("FMatches", len(filtered_matches))
chosen_matches =  filtered_matches[:100]

list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in chosen_matches] 
list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in chosen_matches]

FE = RANSAC_F_matrix(list_kp1,list_kp2)

FE = F
print("Function F matrix: ",FE)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,FE)
lines1 = lines1.reshape(-1,3)
# print(" lines1", lines1)
img5,img6 = drawlines(imA,imB,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,FE)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(imB,imA,lines2,pts2,pts1)
cv.imshow("left",img5)
cv.imshow("right",img3)

E = getEssentialMatrix(K1, K2, FE)

print("Essential matrix : ",E)

cset,rset  = extractCameraPose(E)
print(cset)
print(rset)
w1,h1 = imA.shape
w2,h2 = w1,h1
_, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(list_kp1), np.float32(list_kp2), FE, imgSize=(w1, h1))

# print(" H1 ", H1)
# print(" H2 ", H2)

image_Left_rectified = cv.warpPerspective(imA, H1, (w1, h1))
image_Right_rectified = cv.warpPerspective(imB, H2, (w2, h2))

matched_pts_left_chosen_rectified = cv.perspectiveTransform(np.array(list_kp1).reshape(-1,1,2), H1)
matched_pts_right_chosen_rectified = cv.perspectiveTransform(np.array(list_kp2).reshape(-1,1,2), H2)

H2_T_inv =  np.linalg.inv(H2.T)	
H1_inv = np.linalg.inv(H1)
FM_rectified = np.dot(H2_T_inv, np.dot(FE, H1_inv))
linesL_rectified = cv.computeCorrespondEpilines(matched_pts_left_chosen_rectified,2, FM_rectified)
# print("Lrec: ",linesL_rectified)
linesL_rectified   = linesL_rectified[:,0]
# print("Lrec: ",linesL_rectified)
linesR_rectified = cv.computeCorrespondEpilines(matched_pts_right_chosen_rectified,2, FM_rectified)
linesR_rectified   = linesR_rectified[:,0]

img9,img10 = drawlines(imA,imB,linesL_rectified,pts1,pts2)
img7,img8 = drawlines(imB,imA,linesR_rectified,pts2,pts1)
cv.imshow("Lrectified",img9)
cv.imshow("Rrectifed",img7)

#____________________________Correspondance________________________________
print("Calculating DISPARITY, MAY TAKE 20 TO 25 SECS")


print(" note")
print(" there is a slight error in the code, comment out the depth disparity to view the rectified epilines")

disp,value = compute_disparity(imgA,imgB,blk=5)

time.sleep(1)
# #________________________________Depth______________________________________
baseline1, f1 = 88.39, 1758.23
baseline2, f2 = 221.76, 1742.11
baseline3, f3 = 537.75, 1729.05
params = [(baseline1, f1), (baseline2, f2), (baseline3, f3)]
baseline, f = params[no-1]
depth_map, depth_array = disparity_to_depth(baseline, f, value)
plt.figure(3)
plt.title('Depth Map Grayscale')
plt.imshow(depth_map, cmap='gray')
plt.figure(4)
plt.title('Depth Map Hot')
plt.imshow(depth_map, cmap='hot')
plt.show()

cv.waitKey(0)