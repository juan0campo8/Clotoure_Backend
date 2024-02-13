import sys
import os
import threading
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask import Flask, request
from SAM import SAM_functions as sam


sys.path.append("SAM/SAM_functions.py")
 
app = Flask(__name__)

UPLOAD_FOLDER = os.getcwd() + r'\images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/segment", methods = ['POST'])
def segment_image():
    print("Pinged")
    print(os.getcwd())
    '''
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['upload_folder'], filename))
    '''    
    try:
        if 'file' not in request.files:
            print('No file part')
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return "No selected file"
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            t1 = threading.Thread(target=sam.generate_image, args=(filepath,))
            t1.start()
            return "Creating Image Segmentations"
    except Exception as e:
        return str(e)


