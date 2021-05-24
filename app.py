from Detector import Detector
import io
from flask import Flask, render_template, request, send_from_directory, send_file, session, flash, redirect
from flask_dropzone import Dropzone
from flask_ngrok import run_with_ngrok  # comment if not using colab
from PIL import Image
import requests
import os
import sys
import imghdr
from werkzeug.utils import secure_filename
# for visualizing outputs
import matplotlib.pyplot as plt

import const

ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# =====================================
# configureFlaskApp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_FOLDER'] = './food-recognition/static/uploads'   # The same path in prediction.html !!!
#app.config['MODEL_PATH'] = 'model/mask_rcnn_food-challenge_0026.h5'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "thisisasupersecretkey"
app.config['SECRET_KEY'] = "thisisasupersecretkey"

app.config.update(
    DROPZONE_REDIRECT_VIEW='prediction',  # set redirect view
    DROPZONE_MAX_FILES=20,
)

# creation of a dropzone
dropzone = Dropzone(app)
run_with_ngrok(app)  # comment if not using colab

# instantiation of detector
detector = Detector()

#========================= UTILS FUN ======================================#
def validate_image(stream):
    """
    check if is a valid image
    """
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

# this function is used to predict on the uploaded image


def predict_on_image(uploaded_filepath):
    """
    use the detector to predict the segmentation for a specific image in input.
    save the resulting image in a specific folder
    """
    base_name = remove_ext(uploaded_filepath)
    prediction_path = app.config['UPLOAD_FOLDER'] + "/" + base_name + "_res.jpg"
    result = detector.inference(uploaded_filepath, prediction_path, ) # result is the detection result which contains all detected bboxes. result is a list, and the index corresponds to the category id.
    # fig, ax = plt.subplots(figsize=(16, 16))
    # image = Image.open(prediction_path)
    # ax.imshow(image)
    # plt.show()
    # fig.savefig(prediction_path, bbox_inches='tight')  # save the figure to file
    # plt.close(fig)

    # global model
    # img = Image.open(uploaded_file)
    # img = np.array(img)

    # results = model.detect([img], verbose=0)
    # fig, ax = plt.subplots(figsize=(16, 16))
    # r = results[0]
    # #visualize results and save them to file
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'],figsize=(16,16), ax=ax)
    # fig.savefig('static/prediction.png',bbox_inches='tight')   # save the figure to file
    # plt.close(fig)
    # TODO extract info from results
    response = [{"food": "food_prova", "score": "0.55"}]
    # for p,scr in zip(results[0]['class_ids'],results[0]['scores']):
    #     response.append({"food":class_names[p],"score":str(scr)})
    return response, prediction_path

def remove_ext(filename):
    """
    remove the extension from a filename.
    E.g. imgname.jpg returns imgname
    :param string filename: name of the file
    :return: string
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    return name

def build_relative_path(filename):
    """
    returns the path to a specific file inside the application
    """
    base = os.path.basename(filename)
    rel_path = "static/uploads/"+base
    return rel_path

#========================= APP FUN ======================================#
@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    print("upload files home.......")
    if request.method == "POST":
        print("upload files POST")
        # check if the post request has the file part
        if 'file' not in request.files:
            print("No file path")
            return "No file path", 204
        uploaded_file = request.files['file']
        print("uploaded file:", uploaded_file)
        # if user does not select file, browser also submit an empty part without filename
        if uploaded_file != '':
            print("uploaded file not null")
            filename = secure_filename(uploaded_file.filename) #check if the file is secure
            print("uploaded filename:", filename)
            # check if the extension of the file is correct
            file_ext = os.path.splitext(filename)[1]
            print("File extension:", file_ext)
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
                print("Error invalid extension ", file_ext)
                return "Invalid image", 400
            # if the file extension is correct save the file on disk
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            uploaded_file.save(filepath)
            print("file saved")
            #file_url = url_for('uploaded_file', filename=filename)
            response, prediction_path = predict_on_image(filepath)
            print("response:", response)
            session["response"] = response
            session["filename"] = build_relative_path(filename)
            session["prediction_path"] = build_relative_path(prediction_path)
            return render_template("prediction.html", jsonfile=session["response"],
                                   filepath=session["filename"],
                                   prediction_path=session["prediction_path"])
        else:
            flash('No selected file')
            return redirect(request.url)
    else:
        print("upload files GET")
        return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "GET":
        print("prediction GET")
        return render_template("prediction.html", jsonfile=session["response"],
                               filepath=session["filename"],
                               prediction_path=session["prediction_path"])
    else:
        print("prediction POST")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if uploaded_file != '':
            filename = secure_filename(uploaded_file.filename)  # check if the file is secure
            # check if the extension of the file is correct
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(uploaded_file.stream):
                return "Invalid image", 400
            # if the file extension is correct save the file on disk
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            uploaded_file.save(filepath)
            print("file saved")
            # file_url = url_for('uploaded_file', filename=filename)
            response, prediction_path = predict_on_image(filepath)
            print("response:", response)
            session["response"] = response
            session["filename"] = build_relative_path(filename)
            session["prediction_path"] = build_relative_path(prediction_path)
            return render_template("prediction.html", jsonfile=session["response"],
                                   filepath=session["filename"],
                                   prediction_path=session["prediction_path"])
        else:
            flash('No selected file')
            return "Invalid image", 400


if __name__ == "__main__":
    # # get port. Default to 8080
    # port = int(os.environ.get('PORT', 8080))
    # # run app
    # app.run(host='0.0.0.0', port=port)

    # another method for run
    app.run()
