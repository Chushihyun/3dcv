import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import copy
import random
import math

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

def interpolation(img, x, y):
    #print(img)
    x1 = math.floor(x)
    x2 = math.floor(x+1)
    y1 = math.floor(y)
    y2 = math.floor(y+1)
    x_d1 = x-float(x1)
    x_d2 = float(x2)-x
    y_d1 = y-float(y1)
    y_d2 = float(y2)-y
    img=np.array(img)
    if x1 < 0 or x2 >= img.shape[0] or y1 < 0 or y2 >= img.shape[1]:
        return 0
    else:
        result=0
        result+=x_d1*y_d1*(img[x2][y2].astype(float))
        result+=x_d1*y_d2*(img[x2][y1].astype(float))
        result+=x_d2*y_d1*(img[x1][y2].astype(float))
        result+=x_d2*y_d2*(img[x1][y1].astype(float))
    result.astype(int)
    #print(result)
    return result

def transform(img1, img2,homo_matrix, shape=(-1, -1)):
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
            y = Y[0]
            x = Y[1]
            #y=int((Y[0]))
            #x=int((Y[1]))
            try:
                #new_image[i][j][:]=img1[x][y][:]
                new_image[i][j] = interpolation(img1, x, y)
            except:
                continue
            #print(new_image[i][j])
    return new_image


if __name__ == '__main__':

    # 2-1
    img3 = cv.imread('sys.argv[1]')
    points3 = np.array([[126, 186], [569, 175], [732, 827], [160, 921]])
    # print(points3.shape)
    shape = (1000, 700)
    points4 = np.array(
        [[0, 0], [shape[1]-1, 0], [shape[1]-1, shape[0]-1], [0, shape[0]-1]])
    my_homo_matrix = homography(points3, points4)
    print(my_homo_matrix)

    # 2-2
    new_image = transform(img3,img3, my_homo_matrix, shape=shape)
    cv.imwrite('images/result/2_new.png', new_image)