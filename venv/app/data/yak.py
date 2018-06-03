import cv2
import PIL
import numpy as np
import colorgram
from PIL import Image
from matplotlib import pyplot as plt

hijau = 0
hitam = 0

def treshold(img1, hijau, hitam):
		row, col = img1.shape
		treshold = np.zeros((row,col,1), np.uint8)
		for i in range(0, row):
			for j in range(0, col):
				val = img1[i,j]


				if(val > 15 and val < 65):
					val = 0
					hijau = hijau + 1
				elif (val > 65): val = 255
				else: hitam = hitam + 1


				treshold.itemset((i, j, 0), val)
		return treshold, hijau, hitam

img = cv2.imread("DSC_0322.jpg", 0)
plt.hist(img.ravel(),256,[0,256])
#plt.show()

row, col = img.shape
byka= row*col

img2, hijau, hitam = treshold(img, hijau, hitam)
byk = float(hijau+hitam)/1.0
prop = float(hijau)/byk
hit = float(hitam)/byk
print("hitam: ",float(hit),"prop: ",float(prop),"hijau: ",hijau,"byk: ",float(byk/byka))

# cv2.imshow("img 1", img)
# cv2.imshow("img", img2)


# img = cv2.medianBlur(img, 5)
# th3 = cv2.adaptiveThreshold(img,70,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow(th3)
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr, color = col)
#     plt.xlim([0,256])
# plt.show()
#
# print(histr)


# # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# colors = colorgram.extract('mangga.jpg', 6)
#
# def hitung(image):
#     hijau, hitam = 0
#     row, col, ch= image.shape
#     canvas = np.zeros((row, col, 3), np.uint8)
#     for m in range(0, row):
#         for n in range(0, col):
#             h, s, v = image[m, n]
#            # gray = (red * 0.299 + green * 0.587 + blue * 0.114)
#            # canvas.itemset((m,n,0),gray)
#     return canvas

# print(colors)

cv2.waitKey(0)
cv2.destroyAllWindows()
