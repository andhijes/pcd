import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
import pandas as pd
from pcd import *
from sisdas import Sisdas


#images
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

sisdas  = Sisdas()

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
            link=UPLOAD_FOLDER+filename
            
            #Fungsi PCD
            y,r_final,g_final,b_final,berat = imageProcess(lokasi_file)
            print(y)
            #Fungsi Sidas
            berat,matang    =   Sisdas.klasifikasi(y,r_final,g_final,b_final,berat)
             
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