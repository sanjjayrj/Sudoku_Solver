from flask import Flask, render_template, request, flash, redirect

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        print(request)
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "hello"
        return "reached here"      
    else:
        return render_template('index.html')
        
app.secret_key = 'super secret key'
app.run(debug = True) 