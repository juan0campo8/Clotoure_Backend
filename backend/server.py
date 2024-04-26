import sys
import os
import threading
import datetime
import json
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
#from werkzeug import FileWrapper
from flask import Flask, request, send_file
from flask_cors import CORS
from SAM import SAM_functions as sam
from COS import COS_functions as cos
import io


sys.path.append("SAM/SAM_functions.py")
 
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getcwd() + r'\images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_filename(file, tag):
    return file.filename.rsplit('.', 1)[0] + tag + '.' + file.filename.rsplit('.', 1)[1].lower()

def bytes_file(file):
    f = io.BytesIO()
    f.write(file)
    f.seek(0)
    return f

@app.route("/")
def hello_world():
    return "Hello world"

@app.route("/segment", methods = ['POST'])
def segment_image():
    '''
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['upload_folder'], filename))
    '''    
    try:
        if 'front' and 'back' not in request.files:
            print('Missing file')
            return "Missing file"
        front = request.files['front']
        back = request.files['back']
        if front.filename == '' or back.filename == '':
            print('No selected file')
            return "No selected file"
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
        if (front and allowed_file(front.filename) and (back and allowed_file(back.filename))):
            data = request.form['data']
            DATA = json.loads(data)
            CLOTHING_NAME = DATA['clothing_name']
            ffn = front.filename
            bfn = back.filename
            
            tag = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            CLOTHING_NAME = CLOTHING_NAME + '_' + tag
            front = front.stream.read()
            back = back.stream.read()
            print("Files read")
            #cos.upload_to_folder(CLOTHING_NAME, ffn, 'clotoure', front)
            #cos.upload_to_folder(CLOTHING_NAME, bfn, 'clotoure', back)
            
            '''filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)'''
            # Modify the function for 2 files
            # Spawn 2 threads (front and back)
            # Pass on S3 info (bucket, directory)
            t1 = threading.Thread(target=sam.generate_image, args=(CLOTHING_NAME, bytes_file(front), bytes_file(back) ))
            t1.start()
            # Return a dictionary with the filename + unique tag for future accessibility
            # Add the directory name to the output
            output = {
                'started': True,
                'filetag': CLOTHING_NAME
                }
            
            return json.dumps(output)
    except Exception as e:
        return str(e)

# {TODO} Create get request endpoint for masks (future will be segmentations)
'''
{
    maskfolder: "maskfolder"
}
'''
# 

@app.route("/mult", methods=["POST"])
def man():
    try:
        if 'file' not in request.files:
            print('No file part')
            return "No file uploaded"
        file = request.files['file']
        print(file.filename)
        data = request.form['data']
        DATA = json.loads(data)
        print(DATA)
        return json.loads(data)
    except Exception as e:
        return str(e)

#'''
    
@app.route("/fandb", methods=["POST"])
def fandb():
    try:
        if 'front' and 'back' not in request.files:
            print('Missing file')
            return "Missing file"
        front = request.files['front']
        back = request.files['back']
        if front.filename == '' or back.filename == '':
            print('No selected file')
            return "No selected file"
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
        if (front and allowed_file(front.filename) and (back and allowed_file(back.filename))):
            data = request.form['data']
            DATA = json.loads(data)
            CLOTHING_NAME = DATA['clothing_name']
            ffn = front.filename
            bfn = back.filename
            
            tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            CLOTHING_NAME = CLOTHING_NAME + '_' + str(tag)
            front = front.stream.read()
            back = back.stream.read()
            #cos.upload_to_folder(CLOTHING_NAME, ffn, 'clotoure', front)
            #cos.upload_to_folder(CLOTHING_NAME, bfn, 'clotoure', back)
            

            return('good')
    except Exception as e:
        return str(e)

#'''


@app.route("/segment/manual", methods=["POST"])
def manual_segmentation():
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
            data = request.form['data']
            DATA = json.loads(data)
            INPUT_POINTS = DATA['input_points']
            INPUT_LABEL = DATA['input_label']
            INPUT_BOX = DATA['input_box']
            print("all good")

            tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            #f = file.filename.rsplit('.', 1)[0] + tag
            fn = file.filename.rsplit('.', 1)[0] + tag + '.' + file.filename.rsplit('.', 1)[1].lower()
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)     

            t1 = threading.Thread(target=sam.generate_manual_mask, args=(filepath, fn, INPUT_POINTS, INPUT_LABEL, INPUT_BOX, ))

            t1.start()
            # Return a dictionary with the filename + unique tag for future accessibility
            # Add the directory name to the output
            output = {
                'started': True,
                'filetag': tag
                }
            return(output)
    except Exception as e:
        return str(e)
    

    
@app.route("/get_items", methods=["GET"])
def get_items():
    print('hit')
    try:
        data = cos.get_bucket_items()
        
        output = {
            'items': list(data)
        }
        print(output)
        return json.dumps(output)
        
    except Exception as e:
        print(e)
        return json.dumps(str(e))

@app.route("/get_image", methods=["POSt"])
def get_image():
    try:
        data = request.form['data']
        DATA = json.loads(data)
        FOLDER = DATA['folder']
        file = cos.download_file_bytes(FOLDER)
        return send_file(
        file,
        as_attachment=True,
        download_name= '{folder}.zip',
        mimetype='text/csv'
    )
    except Exception as e:
        print(e)
        return(json.dumps(e))
    
    
'''@app.route("/getmask", methods=["POST"])
def get_masks():
    data = request.form['data']
    DATA = json.loads(data)
    FOLDER = DATA['folder']
    BUCKEt = 'clotoure'
    #masks = FileWrapper(cos.download_file_bytes(FOLDER))'''
    


#'''
