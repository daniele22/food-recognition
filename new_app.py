# import mmdetection
from Detector import Detector
# common lib
import os
import sys
import imghdr
from PIL import Image
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg']
app.config['UPLOAD_PATH'] = '/content/food-recongition/uploads'  # Ricordarsi che è lo stesso path che c'è in prediction.html
#app.config['MODEL_PATH'] = 'model/mask_rcnn_food-challenge_0026.h5'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "thisisasupersecretkey"
app.config['SECRET_KEY'] = "thisisasupersecretkey"

app.config.update(
    UPLOADED_PATH=app.config['UPLOAD_PATH'],
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE=app.config['UPLOAD_EXTENSIONS'],
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
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def predict_on_image(uploaded_file, score_threshold):
    prediction_path = app.config['UPLOAD_PATH'] + "/inference_response.jpg"
    result, final_img = detector.inference(uploaded_file, prediction_path, ) # result is the detection result which contains all detected bboxes. result is a list, and the index corresponds to the category id.
    fig, ax = plt.subplots(figsize=(16, 16))
    image = Image.open(prediction_path)
    ax.imshow(image)
    plt.show()
    response = [{"food": "food_prova", "score": "0.55"}]
    return response


#=================== APP FUN ====================================#
@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def handle_upload():
    # for key, f in request.files.items():
    #     if key.startswith('file'):
    #         f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    # return '', 204
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print("Error file not found!")
            return 'Error file not found', 204
        uploaded_file = request.files['file']
        # if user does not select file, browser also submit an empty part without filename
        if uploaded_file != '':
            filename = secure_filename(uploaded_file.filename) #check if the file is secure
            # check if the extension of the file is correct
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
                return "Invalid image, accepted only jpg images", 400
            # if the file extension is correct save the file on disk
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            uploaded_file.save(filepath)
            print("file saved")
            tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_img")
            os.rename(filepath, tmp_name)
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
    # title = request.form.get('title')
    score_thr = request.form.get('score_threshold')
    print("get score threshold = ", score_thr)
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_img")
    response = predict_on_image(tmp_name, score_threshold=score_thr)
    print("response:", response)
    session["response"] = response
    return render_template("prediction.html", jsonfile=session["response"])
    #return 'file uploaded and form submit<br>title: %s<br> description: %s' % (title, description)

@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template("prediction.html", jsonfile=session["response"])

if __name__ == '__main__':
    app.run()