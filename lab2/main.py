import numpy as np
from PIL import Image, ImageOps
import math
import random

def barymetric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2)) / ((x0 -x2) * (y1-y2) - (x1-x2)*(y0-y2))
    lambda1 = ((x0-x2)*(y-y2)-(x-x2)*(y0-y2)) / ((x0 -x2) * (y1-y2) - (x1-x2)*(y0-y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

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
    color = (int(-255*cos),0,0)
    return color


def draw_triangje(x0, y0, x1, y1, x2,y2, w, h):

    norma = calc_normal(x0,x1,x2,y0,y1,y2,z0,z1,z2)
    cos = visible(norma, [0,0,1])
    color = set_color(cos)
    xmin = float(min(x0,x1,x2))
    ymin = float(min(y0,y1,y2))
    xmax = float(max(x0,x1,x2))
    ymax = float(max(y0,y1,y2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > w -1 ): xmax = w - 1
    if (ymax > h -1 ): ymax = h - 1
    if (cos>0):
        return
    for x in range (int(xmin),int(xmax+1)):
        for y in range(int(ymin), int(ymax + 1)):
            lambda0, lambda1, lambda2 = barymetric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
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
size = len(resultF)
for i in range (0, size):
    x0 = result_array[resultF[i][0]-1][0]*20000+2000
    z0 = result_array[resultF[i][0]-1][2]*20000+2000
    y0 = result_array[resultF[i][0]-1][1]*20000+2000
    x1 = result_array[resultF[i][1]-1][0]*20000+2000
    z1 = result_array[resultF[i][1]-1][2]*20000+2000
    y1 = result_array[resultF[i][1]-1][1]*20000+2000
    x2 = result_array[resultF[i][2]-1][0]*20000+2000
    y2 = result_array[resultF[i][2]-1][1]*20000+2000
    z2 = result_array[resultF[i][2]-1][2]*20000+2000

    draw_triangje(x0, y0, x1, y1, x2, y2, w, h)



img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('lab2_2.png')
