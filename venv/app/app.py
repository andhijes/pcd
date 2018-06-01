import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename

from pcd import *

UPLOAD_FOLDER = 'C:/laragon/www/ki/pcd/venv/app/templates/temp'
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
        # if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            grayscale(filename)
            
            link=UPLOAD_FOLDER+filename
            return render_template('index.html',filename=filename, berhasil="1",value=akurasi,value2=kelas,value3=berat,file_url=link)
    return render_template('index.html',)






