import cv2
import numpy as np
import csv
import glob

#fungsi grayscale
def grayscale(source):
    row, col, ch = source.shape
    graykanvas = np.zeros((row, col, 1), np.uint8)
    for i in range(0, row):
        for j in range(0, col):
            blue, green, red = source[i, j]
            gray = red * 0.299 + green * 0.587 + blue * 0.114
            graykanvas.itemset((i, j, 0), gray)
    return graykanvas

#fungsi untuk mengurangi gambar asli dengan mask
def substract(img, subtractor):
    grey = grayscale(img)
    row, col, ch = img.shape
    canvas = np.zeros((row, col, 3), np.uint8)
    for i in range (0, row):
        for j in range(0, col):
            b, g, r = img[i,j]
            subs = int(grey[i,j]) - int(subtractor[i,j])
            if(subs<0):
                canvas.itemset((i, j, 0), 0)
                canvas.itemset((i, j, 1), 0)
                canvas.itemset((i, j, 2), 0)
            else:
                canvas.itemset((i, j, 0), b)
                canvas.itemset((i, j, 1), g)
                canvas.itemset((i, j, 2), r)
    return canvas

#read all the images
#images = [cv2.imread(file) for file in glob.glob("PCD/*.jpg")]
img = cv2.imread("mangga.jpg")

count = 1
data = []

#buat mask
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)

ret,biner_threshold = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY )

kernel3 = np.ones((9, 9), np.uint8)
dilation3 = cv2.dilate(biner_threshold, kernel3, iterations=15)
erotion3 = cv2.erode(dilation3, kernel3, iterations=15)

biner_threshold = cv2.bitwise_not(erotion3)
final = substract(img, biner_threshold)
final1 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

#inisialisasi diameter vertikal
maks_y = 0
min_y = 99999999999

#inisialisasi luas
berat = 0
full  = 0

#inisialisasi rataan rgb
red = 0
blue = 0
green = 0

r_size = 0
b_size = 0
g_size = 0


row, col = final1.shape
for i in range(0, row):
    for j in range(0, col):
        #nilai dari citra yg telah di mask
        val = final1[i,j]

        #mendapatkan luas dari citra yang telah di mask
        if(val!=0): berat = berat + 1


        #mendapatkan rataan rgb
        b, g, r = final[i,j]

        red = red + r
        green = green + g
        blue = blue + b

        if(r): r_size = r_size + 1
        if(g): g_size = g_size + 1
        if(b): b_size = b_size + 1

        #mendapatkan diameter dari citra yang telah di edge detection
        if (val!=0):
                if(maks_y < j): maks_y = j
                if(min_y > j): min_y = j

#mendapatkan nilai rataan rgb
r_final = float(red)/r_size
g_final = float(green)/g_size
b_final = float(blue)/b_size

#mendapatkan nilai diameter vertikal
y = maks_y - min_y

#mendapatkan nilai luas
full = row*col
berat = float(berat)/full

#print (r_final, g_final, b_final, berat, y)
#data.append([r_final, g_final, b_final, hijau_final, hitam_final, berat])

#myFile = open('lagi4.csv','w')
#with myFile:
#	writer = csv.writer(myFile)
#	writer.writerows(data)
