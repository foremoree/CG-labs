import numpy as np
from PIL import Image, ImageOps
import math
import random

def barymetric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2)) / ((x0 -x2) * (y1-y2) - (x1-x2)*(y0-y2))
    lambda1 = ((x0-x2)*(y-y2)-(x-x2)*(y0-y2)) / ((x0 -x2) * (y1-y2) - (x1-x2)*(y0-y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def X_Strih(alpha):
    return np.array([[1, 0 , 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
def Y_Strih(beta):
    return np.array([[np.cos(beta), 0 , np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
def Z_Strih(gamma):
    return np.array([[np.cos(gamma), np.sin(gamma) , 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

def calc_normal(x0,x1,x2,y0,y1,y2,z0,z1,z2):
    vek1 = [x1-x2,y1-y2,z1-z2]
    vek2 = [x1-x0,y1-y0,z1-z0]
    norma = np.cross(vek1,vek2)
    norma = norma / np.linalg.norm(norma)
    return norma

def visible(norma, l):
    cos = np.dot(norma,l)
    return cos

def set_color(cos):
    color = (int(-167*cos),int(-203*cos),int(-32*cos))
    return color


def draw_triangje(x0, y0, z0, x1, y1, z1, x2, y2, z2, w, h):
    a = 20000*0.2
    px0 = a*x0/z0 + w/2
    py0 = a*y0/z0 + h/2
    px1 = a*x1/z1 + w/2
    py1 = a*y1/z1 + h/2
    px2 = a*x2/z2 + w/2
    py2 = a*y2/z2 + h/2

    norma = calc_normal(x0,x1,x2,y0,y1,y2,z0,z1,z2)
    cos = visible(norma, [0,0,1])
    color = set_color(cos)
    xmin = float(min(px0,px1,px2))
    ymin = float(min(py0,py1,py2))
    xmax = float(max(px0,px1,px2))
    ymax = float(max(py0,py1,py2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > w -1 ): xmax = w - 1
    if (ymax > h -1 ): ymax = h - 1
    if (cos>0):
        return
    for x in range (int(xmin),int(xmax+1)):
        for y in range(int(ymin), int(ymax + 1)):
            lambda0, lambda1, lambda2 = barymetric_coordinates(x, y, px0, py0, px1, py1, px2, py2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                Z = lambda0*z0 + lambda1*z1 + lambda2*z2
                if (Z > z_b[y,x]):
                    continue
                img_mat[y, x] = color
                z_b[y,x] = Z


obj_file = open('model_1.obj')
img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
z_b = np.full((4000, 4000), 10000000.0)
img_size = Image.fromarray(img_mat, mode = 'RGB')
w, h = img_size.im.size

result_array = []
resultF = []
for line in obj_file:
    s = line.split()
    if (s[0] == 'v'):
        numbers = [float(s[1]), float(s[2]), float(s[3])]
        result_array.append(numbers)
    if (s[0] == 'f'):
        numbers2 = [int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])]
        resultF.append(numbers2)
size1 = len(result_array)
alpha = 0
beta = 90
gamma = 0
Tx = 0
Ty = -0.04
Tz = 0.2
Rx = X_Strih(alpha)
Ry = Y_Strih(beta)
Rz = Z_Strih(gamma)
R = np.dot(np.dot(Rx, Ry), Rz)

transform_result_array = []

for j in range(len(result_array)):
    X = result_array[j][0]
    Y = result_array[j][1]
    Z = result_array[j][2]
    vector_XYZ = np.array([X, Y, Z])
    vecor_first = np.dot(R,vector_XYZ) + np.array([Tx,Ty,Tz])
    transform_result_array.append(vecor_first)

size = len(resultF)

for i in range (0, size):
    x0 = transform_result_array[resultF[i][0]-1][0]
    z0 = transform_result_array[resultF[i][0]-1][2]
    y0 = transform_result_array[resultF[i][0]-1][1]
    x1 = transform_result_array[resultF[i][1]-1][0]
    z1 = transform_result_array[resultF[i][1]-1][2]
    y1 = transform_result_array[resultF[i][1]-1][1]
    x2 = transform_result_array[resultF[i][2]-1][0]
    y2 = transform_result_array[resultF[i][2]-1][1]
    z2 = transform_result_array[resultF[i][2]-1][2]
    draw_triangje(x0, y0, z0 ,x1, y1,z1,  x2, y2,z2, w, h)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('lab3_0_2.png')
