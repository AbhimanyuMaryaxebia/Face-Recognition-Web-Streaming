##################################################
# Dependencies
##################################################
import os
import numpy as np
import face_recognition
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tqdm import tqdm
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from fr_packages.load_data import load_trained_data
from fr_packages.find_best_face import find_best_face
from fr_packages.save_dataset import save_user_dataset

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Router 
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image" , methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"status" : "Bad POST request structure."})
        if 'image_type' not in request.form:
            return jsonify({"status" : "Bad POST request structure."})       
        file = request.files['file']
        file_type = str(request.form['image_type'])
        file_type = file_type.lower()
        if file_type == "training":
            app.config['UPLOAD_FOLDER'] = './training_input/'
        elif file_type == "recognize":
            app.config['UPLOAD_FOLDER'] = './user_input/'
        else:
            return jsonify({"status" : "image_type should be training or recognize"})
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({"status" : "POST Request is NULL."})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({"image_name" : filename , "status" : "Success!", "type" : file_type})
    return jsonify({"status" : "POST request is NULL."})


""" MAJOR FUNCTION
For training or re-training single user face
 - user_image_1 : Latest User image with clear face
 - label : Unique Employee username/code for face labelling
 - extension : Default User Image extension is ".jpg" 
"""
@app.route("/trainface", methods=['POST'])
def train_face():
    req_data = request.get_json()
    training_folder = req_data['label']
    path = os.getcwd()
    full_path = path + "/training_input/" + training_folder + "/"
    images = os.listdir(full_path)
    # print(images)
    person_encode = np.zeros(128)
    faces_found = 0
    for image in images:
        if image == ".DS_Store":
            continue
        person_image = face_recognition.load_image_file(full_path + image)
        face_encodings = face_recognition.face_encodings(person_image)
        if len(face_encodings) > 0:
            faces_found += 1
            person_encode += face_encodings[0]
    person_encode[:] = [x / faces_found for x in person_encode]
    print(person_encode)
    trained_user = {training_folder: person_encode}
    save_user_dataset(training_folder, trained_user)
    return jsonify({"status" : f"{training_folder} user_trained"})

@app.route("/train_single_face", methods=['POST'])
def train_single_face():
    req_data = request.get_json()
    print(req_data)
    user_image = req_data['user_image']
    extension = req_data['extension']
    uid = req_data['uid']
    input_image = user_image + extension
    input_path = './training_input/'
    image_matrix = face_recognition.load_image_file(input_path + input_image)
    face_encodings = face_recognition.face_encodings(image_matrix)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        trained_user = {uid: face_encoding}
        save_user_dataset(uid, trained_user)
        return jsonify({"train" : f"{uid} face attributes saved"})
    else:
        return jsonify({"train" : "Failed. Try different image"})

""" MAJOR FUNCTION
For Recognizing faces of input image
 - input_image : Input Image from which face needs to be recognized
 - extension : Default Input Image extension is ".jpg"
 - tolerance : Default tolerance is 0.5 
"""
@app.route("/recognizefaces", methods=['POST'])
def recognize_faces():
    req_data = request.get_json()
    user_image = str(req_data['user_image'])
    tolerance = float(req_data['tolerance'])
    extension = str(req_data['extension'])
    trained_face_encodings = load_trained_data()
    dataset_face_names = list(trained_face_encodings.keys())
    dataset_face_attributes = np.array(list(trained_face_encodings.values()))
    found_faces = []
    input_path = "./user_input/"
    image = face_recognition.load_image_file(input_path + user_image + extension)
    # Get face encodings for all people in the picture
    unknown_face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)
    unknown_face_attributes = face_recognition.face_encodings(image, known_face_locations=unknown_face_locations)
    # PIL Image Font Setup
    pil_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(pil_image)
    fnt = PIL.ImageFont.truetype('arial.ttf', 28)
    i = 0
    # There might be more than one person in the photo, so we need to loop over each face we found
    progressbar = tqdm(unknown_face_attributes, ncols=100)
    for unknown_face_attribute in progressbar:
        # Get Best Match face with under given tolerance
        best_match_name = find_best_face(dataset_face_names, dataset_face_attributes, unknown_face_attributes[i], tolerance)
        top, right, bottom, left = unknown_face_locations[i]
        if best_match_name == "un-identified":
            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 255), width=6)
            draw.rectangle([(left, bottom), (right, bottom + 50)], fill=(255, 0, 0, 0))
            draw.text((left + 5, bottom), best_match_name, font=fnt, fill=(255, 255, 255, 255))
        else:
            found_faces.append(best_match_name)
            draw.rectangle([left, top, right, bottom], outline=(69, 244, 69, 255), width=6)
            draw.rectangle([(left, bottom), (right, bottom + 50)], fill=(69, 244, 69, 255))
            draw.text((left + 5, bottom), best_match_name, font=fnt, fill=(0, 0, 0, 255))
        i = i + 1

    # Display the image on screen
    output_path = "./user_output/"
    pil_image.save(output_path + user_image + "output.jpg")
    print(found_faces)
    if len(found_faces) > 0:
        return jsonify({"recognize" : found_faces, "outputImage": output_path + user_image + "output.jpg"})
    else:
        return jsonify({"recognize" : f"No Faces Found for tolerance {tolerance}"})
    