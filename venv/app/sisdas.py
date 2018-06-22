import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
import pandas as pd


class Sisdas:
    def klasifikasi(y, r_final, g_final, b_final, berat):
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

        luas = berat
        berat = -41.9630 + 3247.8645*luas - 0.0693*y
        berat = '%.3f'%(berat)

        print('hasil berat: ' )
        print(berat)


        return berat,matang   