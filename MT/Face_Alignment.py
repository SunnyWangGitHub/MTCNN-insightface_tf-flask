#coding=utf-8

#             使用仿射变换实现人脸对齐
import cv2
import numpy as np

# coord5point 是变换后的五个关键点的定点坐标
coord5point = np.array([[30.2946, 51.6963],
               [65.5318, 51.5014],
               [48.0252, 71.7366],

               [33.5493, 92.3655],
               [62.7299, 92.2041]])

def transformation_from_points(point1, point2):
    #point1 = point1.astype(np.float64)
    #point2 = point2.astype(np.float64)
    c1 = np.mean(point1, axis = 0)
    c2 = np.mean(point2, axis = 0)

    point1 -= c1
    point2 -= c2

    s1 = np.std(point1)
    s2 = np.std(point2)

    point1 /= s1
    point2 /= s2

    U, S, Vt = np.linalg.svd(point1.T * point2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

def warp_im(img_im, org_landmarks, tar_landmarks):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in org_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))

    M = transformation_from_points(pts1, pts2)

    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))

    return dst

def resizeimage(image_path):

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    size = (int(width * 0.7), int(height * 0.7))
    imgresized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return imgresized