import numpy as np
from PIL import Image, ImageOps
import math
img_mat = np.zeros((4000, 4000, 3), dtype=np.uint8)
obj_file = open('model_1.obj')
def draw_line(x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            if (xchange):
                img_mat[x, y] = color
            else:
                img_mat[y, x] = color
            derror += dy
            if (derror > (x1 - x0)):
                derror -= 2*(x1 - x0)
                y += y_update

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
    y0 = result_array[resultF[i][0]-1][1]*20000+2000
    x1 = result_array[resultF[i][1]-1][0]*20000+2000
    y1 = result_array[resultF[i][1]-1][1]*20000+2000
    x2 = result_array[resultF[i][2]-1][0]*20000+2000
    y2 = result_array[resultF[i][2]-1][1]*20000+2000
    draw_line(x0, y0, x1, y1, [255,124,55])
    draw_line(x1, y1, x2, y2, [194,124,220])
    draw_line(x2, y2, x0, y0, [54,220,130])
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img12.png')


""" Очевидное решение img3.png
def draw_line(x0, y0, x1, y1, color):
    step = 1.0/250
    for t in np.arange (0,1,step):
        x = round((1.0 - t)*x0+t*x1)
        y = round((1.0 - t)*y0+t*y1)
        img_mat[y,x] = color
"""
""" Не выбирая шаг img4.png
def draw_line(x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0/count
    for t in np.arange (0,1,step):
        x = round((1.0 - t)*x0+t*x1)
        y = round((1.0 - t)*y0+t*y1)
        img_mat[y,x] = color
"""
""" Цикл по x img5.png
def draw_line(x0, y0, x1, y1, color):
    for x in range (x0,int(x1)):
        t = (x-x0)/(x1-x0)
        y = round((1.0-t)*y0+t*y1)
        img_mat[y,x] = color
"""
""" Фикс 1 с заменой местами точек img6.png
def draw_line(x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            img_mat[y,x] = color
"""
""" Фикс 2 с заменой оси x->y и y->x 
def draw_line(x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            if (xchange):
                img_mat[x, y] = color
            else:
                img_mat[y, x] = color
"""
""" img8.png
def draw_line(x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = abs(y1 - y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            if (xchange):
                img_mat[x, y] = color
            else:
                img_mat[y, x] = color
            derror += dy
            if (derror > 0.5):
                derror -= 1.0
                y += y_update
"""
""" img9.png
def draw_line(x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2.0*(x1 - x0)*abs(y1 - y0)/(x1-x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            if (xchange):
                img_mat[x, y] = color
            else:
                img_mat[y, x] = color
            derror += dy
            if (derror > 2.0*(x1 - x0)*0.5):
                derror -= 2.0*(x1 - x0)*1.0
                y += y_update
"""
"Адгоритм Брезенхема"
"""
def draw_line(x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range (int(x0), int(x1)):
            t = (x-x0)/(x1-x0)
            y = round((1.0-t)*y0+t*y1)
            if (xchange):
                img_mat[x, y] = color
            else:
                img_mat[y, x] = color
            derror += dy
            if (derror > (x1 - x0)):
                derror -= 2*(x1 - x0)
                y += y_update

for i in range (13):
    x0 = 100
    y0 = 100
    x1 = 100+95*np.cos((i*2*3.14)/13)
    y1 = 100+95*np.sin((i*2*3.14)/13)
    draw_line(x0, y0, x1, y1, 255)
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img10.png')
"""