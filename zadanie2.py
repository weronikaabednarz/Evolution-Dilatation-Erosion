import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = Image.open("picture.bmp")
image2 = Image.open("picture2.bmp")
image3 = Image.open("mapa.bmp")

h,w = image.size

def dylatacja(image):                               #1 operacja morfologiczna

    matrix = np.array(image)
    matrix2 = np.full((w,h), 255, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            for a in range(-1,2):
                for b in range(-1,2):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        if matrix[i+a,j+b]==0:
                            matrix2[i,j] = 0
    img = Image.fromarray(matrix2)
    img.save("1_dylatacja.bmp")
    return img

dylatacja(image)

def erozja(image):                                  #2 operacja morfologiczna
    
    matrix = np.array(image)
    matrix2 = np.full((w,h), 0, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            for a in range(-1,2):
                for b in range(-1,2):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        if matrix[i+a,j+b]==255:
                            matrix2[i,j] = 255
    img = Image.fromarray(matrix2)
    img.save("2_erozja.bmp")
    return img

erozja(image)

def opening(image):                                  #3 operacja morfologiczna
    d = dylatacja(erozja(image))
    d.save("3_dylatacja_erozji.bmp")

opening(image)

def closing(image):                                  #4 operacja morfologiczna
    c = erozja(dylatacja(image))
    c.save("4_erozja_dylatacji.bmp")

closing(image) 

def splot_funkcji(image, maska):                    #obliczanie splotu funkcji z maską o promieniu 1

    matrix = np.array(image)
    matrix2 = np.zeros((w,h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            sum = 0
            for a in range(-1,2):
                for b in range(-1,2):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        sum += maska[a,b] * matrix[i+a,j+b]
            if sum > 255:
                matrix2[i,j]=255
            elif sum < 0:
                matrix2[i,j]=0
            else:
                matrix2[i,j]=sum
    img = Image.fromarray(matrix2)
    img.save("5_splot_funkcji.bmp")
    return img

splot_funkcji(image, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))

def splot_funkcji_r(image, maska, r):               #obliczanie splotu funkcji z maską o promieniu r

    matrix = np.array(image)
    matrix2 = np.zeros((w,h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            sum = 0
            for a in range(-1*r,r+1):
                for b in range(-1*r,r+1):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        sum += maska[a,b] * matrix[i+a,j+b]
            if sum > 255:
                matrix2[i,j]=255
            elif sum < 0:
                matrix2[i,j]=0
            else:
                matrix2[i,j]=sum
    img = Image.fromarray(matrix2)
    img.save("6_splot_funkcji_r.bmp")
    return img

#upperfilter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
upperfilter = np.array([[-1, -1, -1, -1, -1],
                        [-1,  9,  9,  9, -1],
                        [-1,  9,  9,  9, -1],
                        [-1,  9,  9,  9, -1],
                        [-1, -1, -1, -1, -1]])
lowerfilter = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9],
                        [1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9],
                        [1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9],
                        [1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9],
                        [1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
gaussfilter = np.array([[1/256, 4/256, 6/256, 4/256, 1/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [6/256, 24/256, 36/256, 24/256, 6/256],
                        [4/256, 16/256, 24/256, 16/256, 4/256],
                        [1/256, 4/256, 6/256, 4/256, 1/256]])
#gaussfilter = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]) 
maska2 = np.array([[-1,-1,-1,1,-1],[-1,1,-1,-1,1],[-1,9,-1,-1,1],[-1,1,-1,-1,1],[-1,1,-1,-1,1]]) #własny filtr

splot_funkcji_r(image,maska2,3)
#splot_funkcji_r(image,gaussfilter,3)    #nałożenie filtru gaussa

#________________________________________________________________________________________________________
#____________możliwość zmiany promienia sąsiedztwa dla operacji morfologicznych__________________________
#________________________________________________________________________________________________________

def dylatacja_r(image2,r):

    matrix = np.array(image2)
    matrix2 = np.full((w,h), 255, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            for a in range(-1*r,r+1):
                for b in range(-1*r,r+1):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        if matrix[i+a,j+b]==0:
                            matrix2[i,j] = 0
    img = Image.fromarray(matrix2)
    img.save("7_dylatacja_r.bmp")
    return img

dylatacja_r(image2,3)

def erozja_r(image2,r):
    
    matrix = np.array(image2)
    matrix2 = np.full((w,h), 0, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            for a in range(-1*r,r+1):
                for b in range(-1*r,r+1):
                    if i + a >= 0 and j + b >= 0 and i + a <= w - 1 and j + b <= h - 1:
                        if matrix[i+a,j+b]==255:
                            matrix2[i,j] = 255
    img = Image.fromarray(matrix2)
    img.save("8_erozja_r.bmp")
    return img

erozja_r(image2,3)

def opening_r(image2,r):
    d = dylatacja_r(erozja_r(image2,r),r)
    d.save("9_dylatacja_erozji_r.bmp")

opening_r(image2,3)

def closing_r(image2,r):
    c = erozja_r(dylatacja_r(image2,r),r)
    c.save("10_erozja_dylatacji_r.bmp")

closing_r(image2,3) 

#________________________________________________________________________________________________________
#____________wczytywanie predefiniowanych masek/filtrów o stałym/różnym promieniu z pliku________________
#________________________________________________________________________________________________________

def splot_funkcji2(image, maska, nazwa_pliku):                    #obliczanie splotu funkcji z maską o promieniu 1
    matrix = np.array(image)
    matrix2 = np.zeros((w,h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            suma = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if i + a >= 0 and j + b >= 0 and i + a < w and j + b < h:
                        suma += maska[a+1, b+1] * matrix[i+a, j+b]
            if suma > 255:
                matrix2[i, j] = 255
            elif suma < 0:
                matrix2[i, j] = 0
            else:
                matrix2[i, j] = suma
    img = Image.fromarray(matrix2)
    return img


def splot_funkcji_r2(image, maska, r, nazwa_pliku):
    matrix = np.array(image)
    matrix2 = np.zeros((w,h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            suma = 0
            for a in range(-r, r+1):
                for b in range(-r, r+1):
                    if i + a >= 0 and j + b >= 0 and i + a < w and j + b < h:
                        suma += maska[a+r, b+r] * matrix[i+a, j+b]
            if suma > 255:
                matrix2[i, j] = 255
            elif suma < 0:
                matrix2[i, j] = 0
            else:
                matrix2[i, j] = suma
    img = Image.fromarray(matrix2)
    return img


def matrixFromFile(nazwa):
    f = open(nazwa, 'r')
    for i, line in enumerate(f):
        if i == 0:
            size = int(line[0])
            tab = [[0 for x in range(size+1)] for y in range(size)]
        else:
            for j in range(len(line.split())):
                tab[i-1][j] = eval(line.split()[j])
    f.close()
    return tab

mask = np.array(matrixFromFile("gaussfilter.txt"))
c =splot_funkcji2(Image.open("picture.bmp"),mask,mask.shape[0]//2)
plt.imshow(c.convert("RGB"))
c.save("11_splot_dla_gaussfilter.bmp")

mask = np.array(matrixFromFile("lowerfilter.txt"))
c =splot_funkcji2(Image.open("picture.bmp"),mask,mask.shape[0]//2)
plt.imshow(c.convert("RGB"))
c.save("12_splot_dla_lowerfilter.bmp")

mask = np.array(matrixFromFile("maska2.txt"))
c =splot_funkcji2(Image.open("picture.bmp"),mask,mask.shape[0]//2)
plt.imshow(c.convert("RGB"))
c.save("13_splot_dla_maska2.bmp")

mask = np.array(matrixFromFile("upperfilter.txt"))
c =splot_funkcji2(Image.open("picture.bmp"),mask,mask.shape[0]//2)
plt.imshow(c.convert("RGB"))
c.save("14_splot_dla_upperfilter.bmp")


#_________________________________________________________________________________

mask = np.array(matrixFromFile("gaussfilter.txt"))
c =splot_funkcji_r2(Image.open("picture.bmp"),mask,mask.shape[0]//2,3)
plt.imshow(c.convert("RGB"))
c.save("15_splot_r_dla_gaussfilter.bmp")

mask = np.array(matrixFromFile("lowerfilter.txt"))
c =splot_funkcji_r2(Image.open("picture.bmp"),mask,mask.shape[0]//2,3)
plt.imshow(c.convert("RGB"))
c.save("16_splot_r_dla_lowerfilter.bmp")

mask = np.array(matrixFromFile("maska2.txt"))
c =splot_funkcji_r2(Image.open("picture.bmp"),mask,mask.shape[0]//2,3)
plt.imshow(c.convert("RGB"))
c.save("17_splot_r_dla_maska2.bmp")

mask = np.array(matrixFromFile("upperfilter.txt"))
c =splot_funkcji_r2(Image.open("picture.bmp"),mask,mask.shape[0]//2,3)
plt.imshow(c.convert("RGB"))
c.save("18_splot_r_dla_upperfilter.bmp")
