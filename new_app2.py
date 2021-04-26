# import mmdetection
from Detector import Detector
# common lib
import os
import sys
import imghdr
from PIL import Image
import pathlib
# flask libs
from flask import Flask, render_template, request, send_from_directory, send_file, flash, redirect, session
from flask_dropzone import Dropzone
from flask_ngrok import run_with_ngrok  # comment if not using colab
from werkzeug.utils import secure_filename
# for visualizing outputs
import matplotlib.pyplot as plt

basedir = os.path.abspath(os.path.dirname(__file__))

ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
print("PATH: ", pathlib.Path().absolute())
print("root dir: ", ROOT_DIR)
print("basedir: ", basedir)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg']
app.config['UPLOAD_FOLDER'] = './food-recognition/uploads'  # Ricordarsi che è lo stesso path che c'è in prediction.html
#app.config['MODEL_PATH'] = 'model/mask_rcnn_food-challenge_0026.h5'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "thisisasupersecretkey"
app.config['SECRET_KEY'] = "thisisasupersecretkey"

app.config.update(
    UPLOADED_PATH=app.config['UPLOAD_FOLDER'],
    # Flask-Dropzone config:
    #DROPZONE_ALLOWED_FILE_TYPE='image/jpg',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='submit',
)

dropzone = Dropzone(app)
run_with_ngrok(app)  # comment if not using colab

detector = Detector() # create detector instance

#========================= UTILS FUN ======================================#
def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    print("format ", format)
    if not format:
        print("validate return none")
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def predict_on_image(uploaded_file, score_threshold):
    print("predict on image ", uploaded_file)
    prediction_path = app.config['UPLOAD_FOLDER'] + "/inference_response.jpg"
    result, final_img = detector.inference(uploaded_file, prediction_path, score_thr=score_threshold) # result is the detection result which contains all detected bboxes. result is a list, and the index corresponds to the category id.
    fig, ax = plt.subplots(figsize=(16, 16))
    image = Image.open(prediction_path)
    ax.imshow(image)
    plt.show()
    # save the fir in temporary file
    # load the new page!!
    response = [{"food": "food_prova", "score": "0.55"}]
    return response


#=================== APP FUN ====================================#
@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

x = ""
@app.route('/upload', methods=['POST'])
def handle_upload():
    print("handle upload")
    # for key, f in request.files.items():
    #     if key.startswith('file'):
    #         f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    # return '', 204
    if request.method == "POST":
        print("handle uploads POST")
        print(request.files.get('file'))
        # check if the post request has the file part
        found = False
        uploaded_file = ''
        for key, f in request.files.items():
            if key.startswith('file'):
                found = True
                uploaded_file = f
                foud = True
                print("Trovato")
                print("EXISTS PATH (folder): ",os.path.exists(app.config['UPLOAD_FOLDER']))
                print("EXISTS PATH: ",os.path.exists(app.config['UPLOADED_PATH']))
                print("EXISTS PATH (2): ",os.path.exists('./food-recognition/uploads'))
                # for x in os.listdir('./food-recognition'):
                  # print("dir", x)
                print("EXISTS FILE: ", os.path.isfile(os.path.join(app.config['UPLOADED_PATH'], f.filename)))
                f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
                print("EXISTS FILE: ", os.path.isfile(os.path.join(app.config['UPLOADED_PATH'], f.filename)))
                break
                #f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
        if not found:
            print("Error file not found!")
            return 'Error file not found', 204
        #uploaded_file = request.files['file']
        # if user does not select file, browser also submit an empty part without filename
        if uploaded_file != '':
            print("uploaded not null")
            print(uploaded_file)
            print("filename: ", uploaded_file.filename)
            filename = secure_filename(uploaded_file.filename) #check if the file is secure
            print("secure filename", filename)
            # check if the extension of the file is correct
            file_ext = os.path.splitext(filename)[1]
            print("extension ", file_ext)
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] :#or file_ext != validate_image(uploaded_file.stream):
                print("invalid image")
                return "Invalid image, accepted only jpg images", 400
            # if the file extension is correct save the file on disk
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            uploaded_file.save(filepath)
            global x
            x = filepath
            print("file saved")

            #tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_img.jpg")
            #print("EXISTS FILE: ", os.path.isfile(filepath))
            #os.rename(filepath, tmp_name)
            # #file_url = url_for('uploaded_file', filename=filename)
            # response = predict_on_image(filepath, score_threshold=score_thr)
            # print("response:", response)
            # session["response"] = response
            return '', 204
            # return render_template("prediction.html", jsonfile=session["response"])
        else:
            print("Error file not found!")
            return 'Error, No selected file!!!', 204
    else:
        return render_template('index.html')


@app.route('/form', methods=['POST'])
def handle_form():
    print("handle form")
    global x
    print("x = ", x)
    title = request.form.get('title')
    print("title", title)
    score_thr = request.form.get('score_thr_number')
    print("get score threshold = ", score_thr)
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_img.jpg")
    print("temp name", tmp_name)
    response = predict_on_image(x, score_threshold=score_thr)
    print("response:", response)
    session["response"] = response
    return render_template("prediction.html", jsonfile=session["response"])
    #return 'file uploaded and form submit<br>title: %s<br> description: %s' % (title, description)

@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template("prediction.html", jsonfile=session["response"])

if __name__ == '__main__':
    app.run()