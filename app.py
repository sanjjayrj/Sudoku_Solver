from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import os, sys, cv2
import numpy as np
sys.path.insert(1, './bin/')
import sudoku_extractor as sud

UPLOAD_FOLDER = './static/working-dir/'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and len(request.form)==0:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            (board, warped_img) = sud.sudoku_extractor(img_path)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "warped_image.png"), warped_img)
            return render_template('index.html', state=1, arr=board)
    elif request.method == 'POST':
        corrected_board=np.zeros((9,9))
        for elem in request.form:
            index=elem[-2:]
            corrected_board[int(index[0]),int(index[1])]=request.form.get(elem)
        print(corrected_board)
        return render_template('index.html', state=3, arr=None)
    else:
        return render_template('index.html', state=0, arr=None)
        
app.secret_key = 'cb1501f35e034aa18dd6c3743f4363bb'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.run(debug = True) 