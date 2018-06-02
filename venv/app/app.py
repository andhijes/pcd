import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
# import os

# from pcd import *

def grayscale(source):
    row, col, ch = source.shape
    graykanvas = np.zeros((row, col, 1), np.uint8)
    for i in range(0, row):
        for j in range(0, col):
            blue, green, red = source[i, j]
            gray = red * 0.299 + green * 0.587 + blue * 0.114
            graykanvas.itemset((i, j, 0), gray)
    return graykanvas

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


UPLOAD_FOLDER = 'C:/laragon/www/pcd/venv/app/templates/temp'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # utama(filename)

            count = 1
            data = []


            hsv = cv2.cvtColor(filename, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)

            ret,biner_threshold = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY )

            kernel3 = np.ones((9, 9), np.uint8)
            dilation3 = cv2.dilate(biner_threshold, kernel3, iterations=15)
            erotion3 = cv2.erode(dilation3, kernel3, iterations=15)

            # cv2.imshow('gray', erotion3)
            # cv2.imshow('gray1', gray)

            biner_threshold = cv2.bitwise_not(erotion3)
            final = substract(filename, biner_threshold)
            final1 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

            hitam = 0
            hijau = 0
            berat = 0
            full  = 0

            red = 0
            blue = 0
            green = 0

            r_size = 0
            b_size = 0
            g_size = 0

            #proposi RGB
            row, col = final1.shape
            for i in range(0, row):
                for j in range(0, col):
                    val = final1[i,j]
                    b, g, r = final[i,j]
                    #print(b,g,r)

                    #if(g!=0 and r!=0):
                    if (val!=0):
                        #if(b>20): hijau = hijau + 1
                        if(val>15 and val < 65): hijau=hijau+1
                        else: hitam = hitam+1

                    red = red + r
                    green = green + g
                    blue = blue + b

                    if(r): r_size = r_size + 1
                    if(g): g_size = g_size + 1
                    if(b): b_size = b_size + 1

            hijau_final = float(hijau)/(hitam+hijau)
            hitam_final = float(hitam)/(hitam+hijau)
            r_final = float(red)/r_size
            g_final = float(green)/g_size
            b_final = float(blue)/b_size


            berat = hitam+hijau
            full = row*col
            berat = float(berat)/full

            # return r_final, g_final, b_final, hijau_final, hitam_final, berat
            
            link=UPLOAD_FOLDER+filename
            # return render_template('index.html')
            return render_template('index.html',filename=filename,value=r_final,value2=g_final,value3=b_final,value4=hijau_final,value5=hitam_final, value6=berat, file_url=link)
#    r_final, g_final, b_final, hijau_final, hitam_final, berat
    return render_template('index.html',)


# @app.route('/ambil/<filename>', methods=['GET', 'POST'])
# def show_file(filename):
#     return send_from_directory('temp/', filename,as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)