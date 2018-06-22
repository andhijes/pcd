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


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/kematangan',  methods=['GET', 'POST'])
def utama():
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
            lokasi_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(lokasi_file)
            # utama(filename)

            filenamee = filename
            count = 1
            data = []
            
            img = cv2.imread(lokasi_file)
            fix = tuple((300,300))
            img = cv2.resize(img, fix)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)

            ret,biner_threshold = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY )

            kernel3 = np.ones((9, 9), np.uint8)
            dilation3 = cv2.dilate(biner_threshold, kernel3, iterations=15)
            erotion3 = cv2.erode(dilation3, kernel3, iterations=15)

            # cv2.imshow('gray', erotion3)
            # cv2.imshow('gray1', gray)

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

            #proposi RGB
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
            
            link=UPLOAD_FOLDER+filename
            
           
            #KODINGAN SISDAS
            import pandas as pd
            mangga = pd.read_csv('mangga.csv', delimiter=',')


            """## Train Test Split"""

            Xclass = mangga[['r_avg', 'g_avg', 'b_avg']]
            yclass = mangga['Kematangan']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(Xclass, yclass)


            """## Preprocessing"""

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            StandardScaler(copy=True, with_mean=True, with_std=True)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            """## Training Model"""

            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(hidden_layer_sizes=(5,5,5),max_iter=500, activation='logistic', alpha =0.0001, solver='lbfgs', learning_rate='constant',
                                learning_rate_init=0.001)
            mlp.fit(X_train,y_train)

            """## Save Model"""
            import pickle
            # save the model to disk
            filename = 'model_baru.sav'
            pickle.dump(mlp, open(filename, 'wb'))


            """## Buat ngetest klasifikasi"""
            #load model yang udah di save
            import pickle
            mlp = pickle.load(open('model_baru.sav', 'rb'))
            #misal barisnya [[r_avg, g_avg, b_avg]]
            testbaris = [[45.42237509758,46.6865241998439,13.7977100274688]]
            #di praproses
            testbaris = scaler.transform(testbaris)
            #diprediksi
            predictions = mlp.predict(testbaris)
            print("hasil kelas: ")
            #hasil prediksi
            print(predictions)

            matang = predictions 
            if matang == [1]:
                matang = 'Kurang Matang'
            if matang == [2]:
                matang = 'Matang'
            if matang == [3]:
                matang = 'Sangat Matang'


            """## Buat ngetest regresi"""
            #misal input
            # luas = 0.1
            # y = 1
            #model
            luas = berat
            berat = -41.9630 + 3247.8645*luas - 0.0693*y
            #hasil berat
            print('hasil berat: ' )
            print(berat)
            
            


            """## Buat ngetest regresi"""

            # # misal input
            # luas = 0.1
            # y = 1
            # # model
            # berat = -41.9630 + 3247.8645 * luas - 0.0693 * y
            #truncate floating
            berat = '%.3f'%(berat)        
            return render_template('hasil.html',filename=filenamee,tingkat_matang=matang,berat=berat, file_url=link)
    
    return render_template('kematangan.html')




@app.route('/images/<filename>', methods=['GET'])
def show_file(filename):
    return send_from_directory('images/', filename, as_attachment=True)

@app.route('/')
def index():   
    return render_template('index.html',)


if __name__ == "__main__":
    app.run(debug=True)