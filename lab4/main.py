import numpy as np
from PIL import Image, ImageOps
import math
import random
from tqdm import tqdm
import time

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

def set_color(I):
    color = (int(-167*I),int(-203*I),int(-32*I))
    return color


def draw_triangje(x0, y0, z0, x1, y1, z1, x2, y2, z2, w, h, vn0, vn1, vn2, im_matr,wt,ht, v0,u0,v1,u1,v2,u2):
    a = 20000*0.2
    px0 = a*x0/z0 + w/2
    py0 = a*y0/z0 + h/2
    px1 = a*x1/z1 + w/2
    py1 = a*y1/z1 + h/2
    px2 = a*x2/z2 + w/2
    py2 = a*y2/z2 + h/2



    n0 = vertex_normal[vn0]
    n1 = vertex_normal[vn1]
    n2 = vertex_normal[vn2]

    I0 = np.dot(n0, np.array([0, 0, 1]))
    I1 = np.dot(n1, np.array([0, 0, 1]))
    I2 = np.dot(n2, np.array([0, 0, 1]))

    norma = calc_normal(x0,x1,x2,y0,y1,y2,z0,z1,z2)

    xmin = float(min(px0,px1,px2))
    ymin = float(min(py0,py1,py2))
    xmax = float(max(px0,px1,px2))
    ymax = float(max(py0,py1,py2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > w -1 ): xmax = w - 1
    if (ymax > h -1 ): ymax = h - 1

    for x in range (int(xmin),int(xmax+1)):
        for y in range(int(ymin), int(ymax + 1)):
            lambda0, lambda1, lambda2 = barymetric_coordinates(x, y, px0, py0, px1, py1, px2, py2)
            I = (lambda0*I0 + lambda1*I1 + lambda2*I2)
            I = min(0, I)
            colorx = int(wt*(lambda0*u0+lambda1*u1+lambda2*u2))
            colory = int(ht*(lambda0*v0+lambda1*v1+lambda2*v2))
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                Z = lambda0*z0 + lambda1*z1 + lambda2*z2
                if (Z > z_b[y,x]):
                    continue
                img_mat[y, x] = im_matr[colorx,colory]*(-I)
                z_b[y,x] = Z


obj_file = open('model_1.obj')
Text_file = Image.open('bunny-atlas.jpg')
Text_file = ImageOps.flip(Text_file)
im_matr = np.array(Text_file)
wt, ht = 1024,1024
img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
z_b = np.full((4000, 4000), 10000000.0)
img_size = Image.fromarray(img_mat, mode = 'RGB')
w, h = img_size.im.size

result_array = []
resultF = []
resultVT = []
resultTEXTU = []
for line in obj_file:
    s = line.split()
    if (s[0] == 'v'):
        numbers = [float(s[1]), float(s[2]), float(s[3])]
        result_array.append(numbers)
    if (s[0] == 'f'):
        numbers2 = [int(s[1].split('/')[0]), int(s[2].split('/')[0]), int(s[3].split('/')[0])]
        resultF.append(numbers2)
        numbers3 = [int(s[1].split('/')[1]), int(s[2].split('/')[1]), int(s[3].split('/')[1])]
        resultVT.append(numbers3)
    if (s[0] == 'vt'):
        numbers4 = [float(s[1]), float(s[2])]
        resultTEXTU.append(numbers4)

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

vertex_normal = np.zeros((len(result_array),3))
for polig in resultF:
    v0 = polig[0]-1
    v1 = polig[1]-1
    v2 = polig[2]-1

    x0,y0,z0 = result_array[v0]
    x1, y1, z1 = result_array[v1]
    x2, y2, z2 = result_array[v2]

    normalPolig = calc_normal(x0,x1,x2,y0,y1,y2,z0,z1,z2)

    vertex_normal[v0] += normalPolig
    vertex_normal[v1] += normalPolig
    vertex_normal[v2] += normalPolig


vertex_normal = np.dot(R,vertex_normal.T).T
for i in range(len(vertex_normal)):
    norm = np.linalg.norm(vertex_normal[i])
    if norm > 0:
        vertex_normal[i] = vertex_normal[i] / norm














size1 = len(result_array)


transform_result_array = []

for j in range(len(result_array)):
    X = result_array[j][0]
    Y = result_array[j][1]
    Z = result_array[j][2]
    vector_XYZ = np.array([X, Y, Z])
    vecor_first = np.dot(R,vector_XYZ) + np.array([Tx,Ty,Tz])
    transform_result_array.append(vecor_first)

size = len(resultF)

for i in tqdm(range (0, size)):
    x0 = transform_result_array[resultF[i][0]-1][0]
    z0 = transform_result_array[resultF[i][0]-1][2]
    y0 = transform_result_array[resultF[i][0]-1][1]
    x1 = transform_result_array[resultF[i][1]-1][0]
    z1 = transform_result_array[resultF[i][1]-1][2]
    y1 = transform_result_array[resultF[i][1]-1][1]
    x2 = transform_result_array[resultF[i][2]-1][0]
    y2 = transform_result_array[resultF[i][2]-1][1]
    z2 = transform_result_array[resultF[i][2]-1][2]
    vn0 = (resultF[i][0] - 1)
    vn1 = (resultF[i][1] - 1)
    vn2 = (resultF[i][2] - 1)
    v0 = resultTEXTU[resultVT[i][0]-1][0]
    u0 = resultTEXTU[resultVT[i][0]-1][1]
    v1 = resultTEXTU[resultVT[i][1]-1][0]
    u1 = resultTEXTU[resultVT[i][1]-1][1]
    v2 = resultTEXTU[resultVT[i][2]-1][0]
    u2 = resultTEXTU[resultVT[i][2]-1][1]

    draw_triangje(x0, y0, z0 ,x1, y1,z1,  x2, y2,z2, w, h, vn0,vn1,vn2,im_matr,wt,ht, v0,u0,v1,u1,v2,u2)

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('lab3_0_3.png')
