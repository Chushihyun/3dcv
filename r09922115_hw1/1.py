import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import copy
import random
import math


def get_sift_correspondences(img1, img2, num, r):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:num]
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    new_points1, new_points2,combine=ransac(points1,points2,r)
    good_matches=np.array(good_matches)
    cv.namedWindow('match',0)
    cv.resizeWindow('match', 1200, 800)
    img_draw_match = cv.drawMatches(
       img1, kp1, img2, kp2, good_matches[combine], None, flags=cv.DrawMatchesFlags_DEFAULT)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return new_points1, new_points2


def homography(points1, points2):

    matrix = []
    for i in range(points1.shape[0]):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        matrix.append([x1, y1, 1, 0, 0, 0, -x1*x2, -x2*y1, -x2])
        matrix.append([0, 0, 0, x1, y1, 1, -x1*y2, -y2*y1, -y2])
    A = np.array(matrix)
    u, s, v = np.linalg.svd(A)

    v_last = v[-1]
    v_last = v_last/v_last[-1]
    v_last = v_last.reshape((3, 3))

    return v_last


def normalize_points(points):
    # transfer points into mean=0, distance = 1
    mean_0 = np.mean(points[:, 0])
    mean_1 = np.mean(points[:, 1])

    shift = np.array([[1, 0, -1*mean_0], [0, 1, -1*mean_1], [0, 0, 1]])
    scale_rate = np.mean(
        np.sqrt(np.sum((points1[:]-[mean_0, mean_1])**2, axis=1)))
    scale = np.array([[1/scale_rate, 0, 0], [0, 1/scale_rate, 0], [0, 0, 1]])
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    matrix = np.matmul(scale, shift)
    new_points = np.matmul(matrix, points.T).T[:, 0:2]

    return matrix, new_points


def normalized_homography(points1, points2):

    matrix1, nor_points1 = normalize_points(points1)
    matrix2, nor_points2 = normalize_points(points2)
    matrix = homography(nor_points1, nor_points2)
    homo_matrix = np.matmul(np.linalg.inv(matrix2), np.matmul(matrix, matrix1))
    homo_matrix /= homo_matrix[2][2]

    return homo_matrix


def calculate_error(ori, gt, homo_matrix):

    err = 0
    for i in range(ori.shape[0]):
        pre = np.matmul(homo_matrix, [ori[i][0], ori[i][1], 1])
        pre /= pre[-1]
        x = int(pre[0])
        y = int(pre[1])
        err += ((gt[i][0]-x)**2+(gt[i][1]-y)**2)**0.5
        #print(f'{ori[i][0]} , {ori[i][1]} , {x} , {y} , {gt[i][0]} , {gt[i][1]} ')
    err /= ori.shape[0]

    # print(err)
    return err


def interpolation(img, x, y):
    # print(img)

    x1 = math.floor(x)
    x2 = math.floor(x+1)
    y1 = math.floor(y)
    y2 = math.floor(y+1)
    x_d1 = x-float(x1)
    x_d2 = float(x2)-x
    y_d1 = y-float(y1)
    y_d2 = float(y2)-y
    img = np.array(img)
    if x1 < 0 or x2 >= img.shape[0] or y1 < 0 or y2 >= img.shape[1]:
        return 0
    else:
        result = 0
        result += x_d1*y_d1*(img[x2][y2].astype(float))
        result += x_d1*y_d2*(img[x2][y1].astype(float))
        result += x_d2*y_d1*(img[x1][y2].astype(float))
        result += x_d2*y_d2*(img[x1][y1].astype(float))
    result.astype(int)
    # print(result)
    return result


def transform(img1, img2, homo_matrix, shape=(-1, -1)):
    if shape == (-1, -1):
        new_image = np.zeros((img2.shape), dtype='uint8')
    else:
        new_image = np.zeros((shape[0], shape[1], 3), dtype='uint8')

    print(new_image.shape)

    inv_homo = np.linalg.inv(homo_matrix)
    inv_homo = inv_homo/inv_homo[2][2]

    for i in trange(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            Y = np.matmul(inv_homo, [j, i, 1])
            Y = Ｙ/Y[-1]
            #y = Y[0]
            #x = Y[1]
            y=int((Y[0]))
            x=int((Y[1]))
            try:
                new_image[i][j][:]=img1[x][y][:]
                #new_image[i][j] = interpolation(img1, x, y)
            except:
                continue
            # print(new_image[i][j])
    return new_image

def ransac(points1, points2, k=4):
    points1 = np.array(points1)
    points2 = np.array(points2)
    best_combine = []
    times = 300
    best_err = 100000
    for t in trange(times):
        combine = random.sample(range(len(points1)), k)
        p1 = points1[combine]
        p2 = points2[combine]
        homo = homography(p1, p2)
        err = calculate_error(points1, points2, homo)
        if err < best_err:
            best_err = err
            best_combine = combine
    #print(best_combine)

    # img_draw_match = cv.drawMatches(
    #     img1, kp1, img2, kp2, good_matches[best_combine], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imshow('match', img_draw_match)
    # cv.waitKey(0)

    return points1[best_combine], points2[best_combine],best_combine


if __name__ == '__main__':

    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    gt_original = gt_correspondences[0]
    gt_target = gt_correspondences[1]

    # 1-1
    points1, points2= get_sift_correspondences(img1, img2, 1000,4)
    homo_matrix = homography(points1, points2)
    # print(homo_matrix)

    # 1-2
    err = calculate_error(gt_original, gt_target, homo_matrix)
    print(err)

    cv.imwrite('images/result/test.png', transform(img1, img2, homo_matrix))

    # 1-3
    normalized_homo_matrix = normalized_homography(points1, points2)
    nor_err = calculate_error(gt_original, gt_target, normalized_homo_matrix)
    print(normalized_homo_matrix)
    print(nor_err)

    cv.imwrite('images/result/1-2_ransac20_normalized.png',transform(img1, img2, normalized_homo_matrix))
